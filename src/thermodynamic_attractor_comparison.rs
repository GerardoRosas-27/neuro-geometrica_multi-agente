//! Comparación pareada entre el CDT térmico anterior y el motor fasorial.
//!
//! Ambos reciben exactamente el mismo grafo Hebbiano, patrones, cues
//! corrompidos y fases iniciales. La evaluación usa una energía XY común,
//! independiente de los reportes internos de cada arquitectura.

use crate::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
};
use crate::native_rng::{splitmix64, unit_from_u64};
use crate::native_thermodynamic_cdt::{
    NativeCdtEdgeKind, NativeThermoCdtConfig, NativeThermoCdtSubstrate,
};
use num_complex::Complex32;
use std::time::{Duration, Instant};

const EPSILON: f32 = 1.0e-7;

#[derive(Clone, Copy, Debug)]
pub struct AttractorComparisonConfig {
    pub nodes: usize,
    pub patterns: usize,
    pub trials: usize,
    pub corruption_fraction: f32,
    pub phase_jitter: f32,
    pub old_max_steps: usize,
    pub phasor_max_iterations: usize,
    pub phase_residual_tolerance: f32,
    pub attractor_accuracy_threshold: f32,
    pub seed: u64,
}

impl Default for AttractorComparisonConfig {
    fn default() -> Self {
        Self {
            nodes: 64,
            patterns: 4,
            trials: 16,
            corruption_fraction: 0.20,
            phase_jitter: 0.08,
            old_max_steps: 1_500,
            phasor_max_iterations: 400,
            phase_residual_tolerance: 2.0e-3,
            attractor_accuracy_threshold: 0.90,
            seed: 0xA77A_C702_2026,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AttractorArchitectureMetrics {
    pub trials: usize,
    pub converged: usize,
    pub attractor_successes: usize,
    pub total_iterations: usize,
    pub total_energy_evaluations: usize,
    pub elapsed: Duration,
    pub initial_common_energy_sum: f64,
    pub final_common_energy_sum: f64,
    pub phase_residual_sum: f64,
    pub target_accuracy_sum: f64,
}

impl AttractorArchitectureMetrics {
    pub fn convergence_rate(self) -> f64 {
        self.converged as f64 / self.trials.max(1) as f64
    }

    pub fn attractor_success_rate(self) -> f64 {
        self.attractor_successes as f64 / self.trials.max(1) as f64
    }

    pub fn mean_iterations(self) -> f64 {
        self.total_iterations as f64 / self.trials.max(1) as f64
    }

    pub fn mean_energy_evaluations(self) -> f64 {
        self.total_energy_evaluations as f64 / self.trials.max(1) as f64
    }

    pub fn mean_initial_common_energy(self) -> f64 {
        self.initial_common_energy_sum / self.trials.max(1) as f64
    }

    pub fn mean_final_common_energy(self) -> f64 {
        self.final_common_energy_sum / self.trials.max(1) as f64
    }

    pub fn mean_phase_residual(self) -> f64 {
        self.phase_residual_sum / self.trials.max(1) as f64
    }

    pub fn mean_target_accuracy(self) -> f64 {
        self.target_accuracy_sum / self.trials.max(1) as f64
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AttractorComparisonReport {
    pub nodes: usize,
    pub patterns: usize,
    pub trained_edges: usize,
    pub trials: usize,
    pub dataset_checksum: u64,
    pub old_cdt: AttractorArchitectureMetrics,
    pub phasor: AttractorArchitectureMetrics,
}

impl AttractorComparisonReport {
    pub fn wall_time_speedup(self) -> f64 {
        self.old_cdt.elapsed.as_secs_f64() / self.phasor.elapsed.as_secs_f64().max(f64::EPSILON)
    }

    pub fn iteration_reduction(self) -> f64 {
        1.0 - self.phasor.mean_iterations() / self.old_cdt.mean_iterations().max(1.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct TrainedEdge {
    a: usize,
    b: usize,
    weight: f32,
    phase: f32,
}

pub fn compare_thermodynamic_attractors(
    config: AttractorComparisonConfig,
) -> AttractorComparisonReport {
    let config = sanitize_config(config);
    let patterns = walsh_patterns(config.nodes, config.patterns);
    let dataset_checksum = pattern_checksum(&patterns);
    let edges = train_hebbian_edges(&patterns);
    let mut cdt_template = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: 1,
        nodes_per_slice: config.nodes,
        spatial_degree: 1,
        temporal_degree: 1,
        temperature: 0.0,
        dt: 0.08,
        diffusion: 0.0,
        confinement: 0.0,
        pilot_gain: 0.0,
        phase_coupling: 1.0,
        amplitude_decay: 0.0,
        seed: config.seed,
        ..NativeThermoCdtConfig::default()
    });
    cdt_template.replace_edges(edges.iter().map(|edge| {
        (
            edge.a,
            edge.b,
            NativeCdtEdgeKind::Spatial,
            edge.weight,
            edge.phase,
            1.0,
        )
    }));
    cdt_template.temperature.fill(0.0);
    cdt_template.thermal_state.fill(0.0);
    cdt_template.amplitude.fill(1.0);
    cdt_template.activation.fill(0.0);
    let phasor_template = NativePhasorThermodynamicEngine::from_core(
        &cdt_template,
        NativePhasorConfig {
            coupling_strength: 1.0,
            radial_strength: 4.0,
            target_amplitude: 1.0,
            confinement: 0.0,
            stimulus_gain: 0.0,
            entropy_weight: 0.0,
            temperature_scale: 0.0,
            noise_scale: 0.0,
            max_amplitude: 1.5,
            ..NativePhasorConfig::default()
        },
    )
    .expect("el fixture CDT válido debe compilar como motor fasorial");

    let mut old_metrics = AttractorArchitectureMetrics::default();
    let mut phasor_metrics = AttractorArchitectureMetrics::default();
    for trial in 0..config.trials {
        let target_index = trial % patterns.len();
        let target = &patterns[target_index];
        let cue = corrupted_cue(target, config.corruption_fraction, config.seed, trial);
        let initial_phases = cue_phases(
            &cue,
            config.phase_jitter,
            config.seed.rotate_left(17),
            trial,
        );
        let initial_energy = common_phase_observables(&initial_phases, &edges);

        let mut old = cdt_template.clone();
        old.phase.copy_from_slice(&initial_phases);
        old.thermal_state.fill(0.0);
        old.amplitude.fill(1.0);
        let old_started = Instant::now();
        let mut old_steps = 0;
        let mut old_observables = initial_energy;
        for step in 0..config.old_max_steps {
            old.step();
            old_steps = step + 1;
            old_observables = common_phase_observables(&old.phase, &edges);
            if old_observables.residual <= config.phase_residual_tolerance {
                break;
            }
        }
        old_metrics.elapsed += old_started.elapsed();
        let old_accuracy = target_accuracy(&old.phase, target);
        record_metrics(
            &mut old_metrics,
            initial_energy,
            old_observables,
            old_accuracy,
            old_steps,
            old_steps,
            config,
        );

        let mut phasor = phasor_template.clone();
        for (state, phase) in phasor.phasors.iter_mut().zip(&initial_phases) {
            *state = Complex32::from_polar(1.0, *phase);
        }
        let phasor_started = Instant::now();
        let minimization = phasor.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: config.phasor_max_iterations,
            residual_tolerance: config.phase_residual_tolerance,
            topological_warm_start: false,
            ..NativePhasorMinimizerConfig::default()
        });
        phasor_metrics.elapsed += phasor_started.elapsed();
        let phasor_phases = phasor
            .phasors
            .iter()
            .map(|phasor| phasor.arg())
            .collect::<Vec<_>>();
        let phasor_observables = common_phase_observables(&phasor_phases, &edges);
        let phasor_accuracy = target_accuracy(&phasor_phases, target);
        record_metrics(
            &mut phasor_metrics,
            initial_energy,
            phasor_observables,
            phasor_accuracy,
            minimization.iterations,
            minimization.energy_evaluations,
            config,
        );
    }

    AttractorComparisonReport {
        nodes: config.nodes,
        patterns: config.patterns,
        trained_edges: edges.len(),
        trials: config.trials,
        dataset_checksum,
        old_cdt: old_metrics,
        phasor: phasor_metrics,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CommonObservables {
    energy: f32,
    residual: f32,
}

fn record_metrics(
    metrics: &mut AttractorArchitectureMetrics,
    initial: CommonObservables,
    final_observables: CommonObservables,
    accuracy: f32,
    iterations: usize,
    energy_evaluations: usize,
    config: AttractorComparisonConfig,
) {
    metrics.trials += 1;
    metrics.total_iterations += iterations;
    metrics.total_energy_evaluations += energy_evaluations;
    metrics.initial_common_energy_sum += initial.energy as f64;
    metrics.final_common_energy_sum += final_observables.energy as f64;
    metrics.phase_residual_sum += final_observables.residual as f64;
    metrics.target_accuracy_sum += accuracy as f64;
    let converged = final_observables.residual <= config.phase_residual_tolerance;
    metrics.converged += usize::from(converged);
    metrics.attractor_successes +=
        usize::from(converged && accuracy >= config.attractor_accuracy_threshold);
}

fn common_phase_observables(phases: &[f32], edges: &[TrainedEdge]) -> CommonObservables {
    let mut energy = 0.0;
    let mut weight_sum = 0.0;
    let mut gradient = vec![0.0_f32; phases.len()];
    let mut incident_weight = vec![0.0_f32; phases.len()];
    for edge in edges {
        let delta = phases[edge.a] - phases[edge.b] + edge.phase;
        energy += edge.weight * (1.0 - delta.cos());
        weight_sum += edge.weight;
        let flow = edge.weight * delta.sin();
        gradient[edge.a] += flow;
        gradient[edge.b] -= flow;
        incident_weight[edge.a] += edge.weight;
        incident_weight[edge.b] += edge.weight;
    }
    let residual = (gradient
        .iter()
        .zip(&incident_weight)
        .map(|(gradient, weight)| {
            let normalized = *gradient / weight.max(EPSILON);
            normalized * normalized
        })
        .sum::<f32>()
        / phases.len().max(1) as f32)
        .sqrt();
    CommonObservables {
        energy: energy / weight_sum.max(EPSILON),
        residual,
    }
}

fn target_accuracy(phases: &[f32], target: &[i8]) -> f32 {
    let direct = phases
        .iter()
        .zip(target)
        .filter(|(phase, expected)| {
            let observed = if phase.cos() >= 0.0 { 1 } else { -1 };
            observed == **expected
        })
        .count();
    direct.max(target.len() - direct) as f32 / target.len().max(1) as f32
}

fn walsh_patterns(nodes: usize, count: usize) -> Vec<Vec<i8>> {
    (0..count)
        .map(|pattern| {
            let code = pattern;
            (0..nodes)
                .map(|node| {
                    if (node & code).count_ones() % 2 == 0 {
                        1
                    } else {
                        -1
                    }
                })
                .collect()
        })
        .collect()
}

fn train_hebbian_edges(patterns: &[Vec<i8>]) -> Vec<TrainedEdge> {
    let nodes = patterns.first().map_or(0, Vec::len);
    let mut edges = Vec::new();
    for a in 0..nodes {
        for b in (a + 1)..nodes {
            let signed_coupling = patterns
                .iter()
                .map(|pattern| pattern[a] as f32 * pattern[b] as f32)
                .sum::<f32>()
                / nodes.max(1) as f32;
            if signed_coupling.abs() <= EPSILON {
                continue;
            }
            edges.push(TrainedEdge {
                a,
                b,
                weight: signed_coupling.abs(),
                phase: if signed_coupling >= 0.0 {
                    0.0
                } else {
                    std::f32::consts::PI
                },
            });
        }
    }
    edges
}

fn corrupted_cue(target: &[i8], corruption_fraction: f32, seed: u64, trial: usize) -> Vec<i8> {
    let mut ranked = (0..target.len())
        .map(|node| {
            (
                splitmix64(seed ^ (trial as u64).rotate_left(23) ^ node as u64),
                node,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    let corrupt = ((target.len() as f32 * corruption_fraction).round() as usize)
        .clamp(1, target.len().saturating_sub(1));
    let mut cue = target.to_vec();
    for (_, node) in ranked.into_iter().take(corrupt) {
        cue[node] *= -1;
    }
    cue
}

fn cue_phases(cue: &[i8], jitter: f32, seed: u64, trial: usize) -> Vec<f32> {
    cue.iter()
        .enumerate()
        .map(|(node, bit)| {
            let base = if *bit > 0 { 0.0 } else { std::f32::consts::PI };
            let unit = unit_from_u64(splitmix64(
                seed ^ (trial as u64).rotate_left(31) ^ node as u64,
            ));
            (base + jitter * (2.0 * unit - 1.0)).rem_euclid(std::f32::consts::TAU)
        })
        .collect()
}

fn pattern_checksum(patterns: &[Vec<i8>]) -> u64 {
    patterns
        .iter()
        .flatten()
        .enumerate()
        .fold(0xCBF2_9CE4_8422_2325, |hash, (index, value)| {
            splitmix64(hash ^ index as u64 ^ (*value as i64 as u64))
        })
}

fn sanitize_config(config: AttractorComparisonConfig) -> AttractorComparisonConfig {
    AttractorComparisonConfig {
        nodes: config.nodes.max(8).next_power_of_two(),
        patterns: config.patterns.clamp(1, config.nodes.max(8)),
        trials: config.trials.max(1),
        corruption_fraction: config.corruption_fraction.clamp(0.01, 0.49),
        phase_jitter: config.phase_jitter.clamp(1.0e-3, 0.5),
        old_max_steps: config.old_max_steps.max(1),
        phasor_max_iterations: config.phasor_max_iterations.max(1),
        phase_residual_tolerance: config.phase_residual_tolerance.max(1.0e-6),
        attractor_accuracy_threshold: config.attractor_accuracy_threshold.clamp(0.5, 1.0),
        ..config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_synthetic_dataset_forms_measurable_attractors() {
        let report = compare_thermodynamic_attractors(AttractorComparisonConfig {
            nodes: 32,
            patterns: 4,
            trials: 8,
            old_max_steps: 800,
            phasor_max_iterations: 250,
            ..AttractorComparisonConfig::default()
        });
        assert_ne!(report.dataset_checksum, 0);
        assert!(report.trained_edges > 0);
        assert_eq!(
            report.old_cdt.mean_initial_common_energy(),
            report.phasor.mean_initial_common_energy()
        );
        assert!(
            report.phasor.mean_final_common_energy() < report.phasor.mean_initial_common_energy(),
            "{report:?}"
        );
        assert!(
            report.old_cdt.mean_final_common_energy() < report.old_cdt.mean_initial_common_energy(),
            "{report:?}"
        );
        assert!(
            report.old_cdt.attractor_success_rate() >= 0.75,
            "{report:?}"
        );
        assert!(report.phasor.attractor_success_rate() >= 0.75, "{report:?}");
        assert!(
            report.phasor.mean_final_common_energy()
                <= report.old_cdt.mean_final_common_energy() * 1.05,
            "{report:?}"
        );
        assert!(
            report.phasor.mean_iterations() < report.old_cdt.mean_iterations(),
            "{report:?}"
        );
    }
}
