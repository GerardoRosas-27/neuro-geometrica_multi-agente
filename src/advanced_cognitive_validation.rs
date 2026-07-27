//! Pruebas adversariales de selección ambigua y descubrimiento de simetría.
//!
//! La simetría descubierta se restringe al grupo de traslaciones cíclicas de
//! canales. El algoritmo recibe ejemplos entrada/salida, pero no la órbita ni
//! el desplazamiento. Esto prueba descubrimiento estructural limitado, no
//! inducción arbitraria de simetrías.

use crate::matrix_free_cognitive_substrate::{LatentConceptId, SparsePattern};
use crate::relational_field::ObserverId;
use crate::unified_spin_cognitive_engine::{
    UnifiedSpinCognitiveConfig, UnifiedSpinCognitiveEngine,
};
use serde::Serialize;

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug)]
pub struct StructuralExample {
    pub source: SparsePattern,
    pub target: SparsePattern,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct CyclicSymmetryDiscovery {
    pub channels: usize,
    pub examples: usize,
    pub shift: usize,
    pub reconstruction_error: f64,
    pub alternative_error: f64,
    pub error_margin: f64,
    pub confidence: f64,
    pub accepted: bool,
}

impl CyclicSymmetryDiscovery {
    pub fn predict(&self, source: &SparsePattern) -> Option<SparsePattern> {
        self.accepted
            .then(|| shift_pattern(source, self.channels, self.shift))
    }
}

pub fn discover_cyclic_symmetry(
    examples: &[StructuralExample],
    channels: usize,
    minimum_examples: usize,
    maximum_error: f64,
    minimum_error_margin: f64,
) -> CyclicSymmetryDiscovery {
    if channels < 2 || examples.len() < minimum_examples.max(2) {
        return CyclicSymmetryDiscovery {
            channels,
            examples: examples.len(),
            ..CyclicSymmetryDiscovery::default()
        };
    }
    let mut ranked = (0..channels)
        .map(|shift| {
            let mut errors = examples
                .iter()
                .map(|example| {
                    pattern_error(
                        &shift_pattern(&example.source, channels, shift),
                        &example.target,
                        channels,
                    )
                })
                .collect::<Vec<_>>();
            errors.sort_by(f64::total_cmp);
            let middle = errors.len() / 2;
            let error = if errors.len() % 2 == 0 {
                0.5 * (errors[middle - 1] + errors[middle])
            } else {
                errors[middle]
            };
            (error, shift)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let (reconstruction_error, shift) = ranked[0];
    let alternative_error = ranked[1].0;
    let error_margin = (alternative_error - reconstruction_error).max(0.0);
    let accepted = reconstruction_error <= maximum_error.max(0.0)
        && error_margin >= minimum_error_margin.max(0.0);
    CyclicSymmetryDiscovery {
        channels,
        examples: examples.len(),
        shift,
        reconstruction_error,
        alternative_error,
        error_margin,
        confidence: if accepted {
            (1.0 - reconstruction_error / alternative_error.max(EPSILON)).clamp(0.0, 1.0)
        } else {
            0.0
        },
        accepted,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdvancedCognitiveValidationConfig {
    pub trials: usize,
    pub minimum_rate: f64,
    pub ambiguity_margin: f64,
}

impl Default for AdvancedCognitiveValidationConfig {
    fn default() -> Self {
        Self {
            trials: 36,
            minimum_rate: 0.90,
            ambiguity_margin: 0.06,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct AdvancedCognitiveValidationReport {
    pub trials: usize,
    pub topology_scales: usize,
    pub branch_selection_accuracy: f64,
    pub trajectory_selection_accuracy: f64,
    pub ambiguity_abstention_rate: f64,
    pub selected_energy_order_accuracy: f64,
    pub mean_selected_margin: f64,
    pub mean_ambiguous_margin: f64,
    pub autonomous_symmetry_discovery_rate: f64,
    pub heldout_structural_transfer_rate: f64,
    pub conflicting_structure_rejection_rate: f64,
    pub decision: &'static str,
}

pub fn run_advanced_cognitive_validation(
    config: AdvancedCognitiveValidationConfig,
) -> AdvancedCognitiveValidationReport {
    let trials = config.trials.max(1);
    let minimum_rate = config.minimum_rate.clamp(0.0, 1.0);
    let mut branch_successes = 0;
    let mut branch_queries = 0;
    let mut trajectory_successes = 0;
    let mut ambiguity_successes = 0;
    let mut ambiguity_queries = 0;
    let mut energy_order_successes = 0;
    let mut selected_margin_sum = 0.0;
    let mut ambiguous_margin_sum = 0.0;
    let mut discovery_successes = 0;
    let mut transfer_successes = 0;
    let mut conflict_rejections = 0;

    for trial in 0..trials {
        let observer = ObserverId(1_400_000 + trial);
        let base_phase = (0.23 + trial as f64 * 0.173).rem_euclid(std::f64::consts::TAU);
        let mut engine = ambiguity_fixture(1 + trial % 3);
        let exposures = 18 + trial % 9;
        engine.train_relation(
            observer,
            LatentConceptId(0),
            LatentConceptId(1),
            base_phase,
            1.0,
            0.0,
            &[],
            exposures,
        );
        engine.train_relation(
            observer,
            LatentConceptId(1),
            LatentConceptId(2),
            base_phase,
            1.0,
            0.0,
            &[],
            exposures,
        );
        engine.train_relation(
            observer,
            LatentConceptId(0),
            LatentConceptId(3),
            base_phase + std::f64::consts::PI,
            1.0,
            0.0,
            &[],
            exposures,
        );

        for jitter in [-0.12, 0.0, 0.12] {
            let left = engine.cognition.workspace.query_with_ambiguity(
                observer,
                LatentConceptId(0),
                base_phase + jitter,
                config.ambiguity_margin,
            );
            branch_successes +=
                usize::from(left.selected == Some(LatentConceptId(1)) && !left.ambiguous);
            energy_order_successes += usize::from(
                left.hypotheses.len() >= 2
                    && left.hypotheses[0].effective_energy < left.hypotheses[1].effective_energy,
            );
            selected_margin_sum += left.margin;
            branch_queries += 1;

            let right = engine.cognition.workspace.query_with_ambiguity(
                observer,
                LatentConceptId(0),
                base_phase + std::f64::consts::PI + jitter,
                config.ambiguity_margin,
            );
            branch_successes +=
                usize::from(right.selected == Some(LatentConceptId(3)) && !right.ambiguous);
            energy_order_successes += usize::from(
                right.hypotheses.len() >= 2
                    && right.hypotheses[0].effective_energy < right.hypotheses[1].effective_energy,
            );
            selected_margin_sum += right.margin;
            branch_queries += 1;
        }
        trajectory_successes += usize::from(
            engine
                .infer(observer, LatentConceptId(0), base_phase, 2)
                .is_some_and(|inference| {
                    inference.path
                        == vec![LatentConceptId(0), LatentConceptId(1), LatentConceptId(2)]
                }),
        );
        for offset in [-0.03, 0.0, 0.03] {
            let ambiguous = engine.cognition.workspace.query_with_ambiguity(
                observer,
                LatentConceptId(0),
                base_phase + std::f64::consts::FRAC_PI_2 + offset,
                config.ambiguity_margin,
            );
            ambiguity_successes += usize::from(ambiguous.ambiguous && ambiguous.selected.is_none());
            ambiguous_margin_sum += ambiguous.margin;
            ambiguity_queries += 1;
        }

        let channels = [8, 12, 16][trial % 3];
        let shift = 1 + (splitmix64(0x51A7_0000 ^ trial as u64) as usize % (channels - 1));
        let first = generated_pattern(channels, trial as u64 * 5 + 1);
        let second = generated_pattern(channels, trial as u64 * 5 + 2);
        let third = generated_pattern(channels, trial as u64 * 5 + 3);
        let outlier = generated_pattern(channels, trial as u64 * 5 + 4);
        let heldout = generated_pattern(channels, trial as u64 * 5 + 5);
        let conflicting_shift = shift % (channels - 1) + 1;
        let examples = [
            StructuralExample {
                source: first.clone(),
                target: noisy_shifted_pattern(&first, channels, shift, trial as u64 * 7 + 1),
            },
            StructuralExample {
                source: second.clone(),
                target: noisy_shifted_pattern(&second, channels, shift, trial as u64 * 7 + 2),
            },
            StructuralExample {
                source: third.clone(),
                target: noisy_shifted_pattern(&third, channels, shift, trial as u64 * 7 + 3),
            },
            StructuralExample {
                source: outlier.clone(),
                target: noisy_shifted_pattern(
                    &outlier,
                    channels,
                    conflicting_shift,
                    trial as u64 * 7 + 4,
                ),
            },
        ];
        let discovery = discover_cyclic_symmetry(&examples, channels, 3, 5.0e-4, 1.0e-3);
        discovery_successes += usize::from(discovery.accepted && discovery.shift == shift);
        transfer_successes += usize::from(discovery.predict(&heldout).is_some_and(|prediction| {
            pattern_error(
                &prediction,
                &shift_pattern(&heldout, channels, shift),
                channels,
            ) <= 1.0e-12
        }));

        let conflicting_examples = [
            StructuralExample {
                source: first.clone(),
                target: shift_pattern(&first, channels, shift),
            },
            StructuralExample {
                source: second.clone(),
                target: shift_pattern(&second, channels, conflicting_shift),
            },
        ];
        let conflict =
            discover_cyclic_symmetry(&conflicting_examples, channels, 2, 1.0e-12, 1.0e-3);
        conflict_rejections += usize::from(!conflict.accepted);
    }

    let branch_denominator = branch_queries.max(1) as f64;
    let ambiguity_denominator = ambiguity_queries.max(1) as f64;
    let trial_denominator = trials as f64;
    let mut report = AdvancedCognitiveValidationReport {
        trials,
        topology_scales: 3,
        branch_selection_accuracy: branch_successes as f64 / branch_denominator,
        trajectory_selection_accuracy: trajectory_successes as f64 / trial_denominator,
        ambiguity_abstention_rate: ambiguity_successes as f64 / ambiguity_denominator,
        selected_energy_order_accuracy: energy_order_successes as f64 / branch_denominator,
        mean_selected_margin: selected_margin_sum / branch_denominator,
        mean_ambiguous_margin: ambiguous_margin_sum / ambiguity_denominator,
        autonomous_symmetry_discovery_rate: discovery_successes as f64 / trial_denominator,
        heldout_structural_transfer_rate: transfer_successes as f64 / trial_denominator,
        conflicting_structure_rejection_rate: conflict_rejections as f64 / trial_denominator,
        decision: "advanced_cognitive_validation_not_demonstrated",
    };
    if report.branch_selection_accuracy >= minimum_rate
        && report.trajectory_selection_accuracy >= minimum_rate
        && report.ambiguity_abstention_rate >= minimum_rate
        && report.selected_energy_order_accuracy >= minimum_rate
        && report.autonomous_symmetry_discovery_rate >= minimum_rate
        && report.heldout_structural_transfer_rate >= minimum_rate
        && report.conflicting_structure_rejection_rate >= minimum_rate
        && report.mean_selected_margin > report.mean_ambiguous_margin
    {
        report.decision = "adversarial_selection_and_limited_symmetry_discovery_pass";
    }
    report
}

fn ambiguity_fixture(scale: usize) -> UnifiedSpinCognitiveEngine {
    UnifiedSpinCognitiveEngine::periodic_pyrochlore(
        scale,
        1,
        1,
        UnifiedSpinCognitiveConfig {
            bootstrap_cooling_steps: 120,
            cooling_steps_per_observation: 1,
            real_steps_per_observation: 0,
            ..UnifiedSpinCognitiveConfig::default()
        },
    )
    .expect("el fixture cognitivo adversarial debe ser válido")
}

fn generated_pattern(channels: usize, seed: u64) -> SparsePattern {
    let first = splitmix64(seed ^ 0xA11C_E001) as usize % channels;
    let mut second = splitmix64(seed ^ 0xA11C_E002) as usize % channels;
    let mut third = splitmix64(seed ^ 0xA11C_E003) as usize % channels;
    if second == first {
        second = (second + 1) % channels;
    }
    while third == first || third == second {
        third = (third + 1) % channels;
    }
    SparsePattern::new([(first, 1.0), (second, 0.57), (third, -0.31)])
}

fn shift_pattern(pattern: &SparsePattern, channels: usize, shift: usize) -> SparsePattern {
    SparsePattern::new(
        pattern
            .components
            .iter()
            .map(|&(channel, value)| ((channel + shift) % channels, value)),
    )
}

fn noisy_shifted_pattern(
    pattern: &SparsePattern,
    channels: usize,
    shift: usize,
    seed: u64,
) -> SparsePattern {
    SparsePattern::new(
        pattern
            .components
            .iter()
            .enumerate()
            .map(|(index, &(channel, value))| {
                let unit = unit_from_u64(splitmix64(seed ^ index as u64));
                (
                    (channel + shift) % channels,
                    value * (1.0 + 0.04 * (2.0 * unit - 1.0)),
                )
            }),
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

#[inline(always)]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[inline(always)]
fn unit_from_u64(value: u64) -> f64 {
    ((value >> 40) as f64) * (1.0 / (1_u64 << 24) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adversarial_branching_and_symmetry_discovery_pass_multiscale_gate() {
        let report = run_advanced_cognitive_validation(AdvancedCognitiveValidationConfig {
            trials: 9,
            ..AdvancedCognitiveValidationConfig::default()
        });
        assert_eq!(
            report.decision, "adversarial_selection_and_limited_symmetry_discovery_pass",
            "{report:#?}"
        );
    }
}
