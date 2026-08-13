//! Comparación pareada: arquitectura legacy (híbrido fasorial-CDT directo)
//! vs arquitectura nueva (RFF fasorial + Softmax CTP + reservorio Langevin).
//!
//! Ambas reciben la misma secuencia sintética y pares plantados de apretón de
//! manos. Las métricas comparables son alineación, consolidación CDT, coste y
//! calidad del handshake.

use crate::hybrid_thermo_attention::{
    HybridThermoAttention, HybridThermoAttentionConfig, PhasorRffConfig,
};
use crate::native_hybrid_phasor_cdt_engine::{
    NativeHybridConfig, NativeHybridPhasorCdtEngine, NativePhasorCue,
};
use crate::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorMinimizerConfig,
};
use crate::native_rng::{splitmix64, signed_unit};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use std::time::{Duration, Instant};

const EPSILON: f32 = 1.0e-7;

// ── Configuración ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct HybridLegacyComparisonConfig {
    pub sequence_length: usize,
    pub d_model: usize,
    pub d_v: usize,
    pub planted_pairs: usize,
    pub trials: usize,
    pub cdt_nodes: usize,
    pub seed: u64,
}

impl Default for HybridLegacyComparisonConfig {
    fn default() -> Self {
        Self {
            sequence_length: 16,
            d_model: 16,
            d_v: 8,
            planted_pairs: 4,
            trials: 12,
            cdt_nodes: 128,
            seed: 0x4C45_4741_4359_2026,
        }
    }
}

// ── Métricas ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct HandshakeMetrics {
    pub top1_rate: f32,
    pub mean_reciprocal_rank: f32,
    pub planted_mass: f32,
    pub softmax_entropy: f32,
    pub rff_softmax_correlation: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EfficiencyMetrics {
    pub wall_time: Duration,
    pub minimizer_iterations: f32,
    pub energy_evaluations: f32,
    pub reservoir_features: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ConsolidationMetrics {
    pub wake_gate_passed: bool,
    pub sleep_accepted: usize,
    pub sleep_rejected: usize,
    pub recall_coherence: f32,
    pub recall_alignment: f32,
    pub thermo_free_energy: f32,
    pub ctp_bias_norm: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ArchitectureTrialMetrics {
    pub handshake: HandshakeMetrics,
    pub efficiency: EfficiencyMetrics,
    pub consolidation: ConsolidationMetrics,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ArchitectureAggregateMetrics {
    pub trials: usize,
    pub handshake_top1_sum: f64,
    pub handshake_mrr_sum: f64,
    pub handshake_planted_mass_sum: f64,
    pub handshake_entropy_sum: f64,
    pub rff_correlation_sum: f64,
    pub wall_time: Duration,
    pub minimizer_iterations_sum: f64,
    pub energy_evaluations_sum: f64,
    pub wake_pass_count: usize,
    pub sleep_accepted_sum: usize,
    pub recall_coherence_sum: f64,
    pub recall_alignment_sum: f64,
    pub thermo_free_energy_sum: f64,
    pub ctp_bias_norm_sum: f64,
}

impl ArchitectureAggregateMetrics {
    pub fn record(&mut self, trial: ArchitectureTrialMetrics) {
        self.trials += 1;
        self.handshake_top1_sum += f64::from(trial.handshake.top1_rate);
        self.handshake_mrr_sum += f64::from(trial.handshake.mean_reciprocal_rank);
        self.handshake_planted_mass_sum += f64::from(trial.handshake.planted_mass);
        self.handshake_entropy_sum += f64::from(trial.handshake.softmax_entropy);
        self.rff_correlation_sum += f64::from(trial.handshake.rff_softmax_correlation);
        self.wall_time += trial.efficiency.wall_time;
        self.minimizer_iterations_sum += f64::from(trial.efficiency.minimizer_iterations);
        self.energy_evaluations_sum += f64::from(trial.efficiency.energy_evaluations);
        self.wake_pass_count += usize::from(trial.consolidation.wake_gate_passed);
        self.sleep_accepted_sum += trial.consolidation.sleep_accepted;
        self.recall_coherence_sum += f64::from(trial.consolidation.recall_coherence);
        self.recall_alignment_sum += f64::from(trial.consolidation.recall_alignment);
        self.thermo_free_energy_sum += f64::from(trial.consolidation.thermo_free_energy);
        self.ctp_bias_norm_sum += f64::from(trial.consolidation.ctp_bias_norm);
    }

    pub fn mean_top1(&self) -> f64 {
        self.handshake_top1_sum / self.trials.max(1) as f64
    }

    pub fn mean_mrr(&self) -> f64 {
        self.handshake_mrr_sum / self.trials.max(1) as f64
    }

    pub fn mean_planted_mass(&self) -> f64 {
        self.handshake_planted_mass_sum / self.trials.max(1) as f64
    }

    pub fn mean_entropy(&self) -> f64 {
        self.handshake_entropy_sum / self.trials.max(1) as f64
    }

    pub fn mean_rff_correlation(&self) -> f64 {
        self.rff_correlation_sum / self.trials.max(1) as f64
    }

    pub fn mean_wall_ms(&self) -> f64 {
        self.wall_time.as_secs_f64() * 1_000.0 / self.trials.max(1) as f64
    }

    pub fn mean_iterations(&self) -> f64 {
        self.minimizer_iterations_sum / self.trials.max(1) as f64
    }

    pub fn wake_pass_rate(&self) -> f64 {
        self.wake_pass_count as f64 / self.trials.max(1) as f64
    }

    pub fn mean_sleep_accepted(&self) -> f64 {
        self.sleep_accepted_sum as f64 / self.trials.max(1) as f64
    }

    pub fn mean_recall_coherence(&self) -> f64 {
        self.recall_coherence_sum / self.trials.max(1) as f64
    }

    pub fn mean_recall_alignment(&self) -> f64 {
        self.recall_alignment_sum / self.trials.max(1) as f64
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HybridLegacyComparisonReport {
    pub config: HybridLegacyComparisonConfig,
    pub legacy: ArchitectureAggregateMetrics,
    pub hybrid: ArchitectureAggregateMetrics,
}

impl HybridLegacyComparisonReport {
    pub fn handshake_top1_delta(&self) -> f64 {
        self.hybrid.mean_top1() - self.legacy.mean_top1()
    }

    pub fn handshake_mrr_delta(&self) -> f64 {
        self.hybrid.mean_mrr() - self.legacy.mean_mrr()
    }

    pub fn wall_time_ratio(&self) -> f64 {
        self.hybrid.mean_wall_ms() / self.legacy.mean_wall_ms().max(f64::EPSILON)
    }

    pub fn sleep_accepted_delta(&self) -> f64 {
        self.hybrid.mean_sleep_accepted() - self.legacy.mean_sleep_accepted()
    }

    pub fn recall_coherence_delta(&self) -> f64 {
        self.hybrid.mean_recall_coherence() - self.legacy.mean_recall_coherence()
    }
}

// ── Ejecución pareada ────────────────────────────────────────────────────────

pub fn run_hybrid_legacy_comparison(
    config: HybridLegacyComparisonConfig,
) -> HybridLegacyComparisonReport {
    let mut legacy = ArchitectureAggregateMetrics::default();
    let mut hybrid = ArchitectureAggregateMetrics::default();

    for trial in 0..config.trials {
        let seed = config.seed ^ (trial as u64).rotate_left(29);
        let (tokens, values, pairs) = planted_handshake_sequence(
            config.sequence_length,
            config.d_model,
            config.d_v,
            config.planted_pairs,
            seed,
        );

        legacy.record(run_legacy_trial(&config, &tokens, &values, &pairs, seed));
        hybrid.record(run_hybrid_trial(&config, &tokens, &values, &pairs, seed));
    }

    HybridLegacyComparisonReport {
        config,
        legacy,
        hybrid,
    }
}

fn run_legacy_trial(
    config: &HybridLegacyComparisonConfig,
    tokens: &[Vec<f32>],
    _values: &[Vec<f32>],
    pairs: &[(usize, usize)],
    seed: u64,
) -> ArchitectureTrialMetrics {
    let hybrid_config = shared_hybrid_config(config);
    let mut engine = NativeHybridPhasorCdtEngine::new(
        NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: config.cdt_nodes,
            spatial_degree: 4,
            temporal_degree: 1,
            temperature: 0.0,
            seed,
            ..NativeThermoCdtConfig::default()
        },
        NativePhasorConfig {
            temperature_scale: 0.02,
            noise_scale: 0.0,
            ..NativePhasorConfig::default()
        },
        hybrid_config,
    )
    .expect("legacy engine");

    let cues = tokens_to_legacy_cues(tokens, engine.core.node_count(), seed);
    let started = Instant::now();
    let wake = engine.infer_and_stage(&cues).expect("legacy wake");
    let wake_elapsed = started.elapsed();

    let recall_cues = tokens_to_legacy_cues(&[tokens[0].clone()], engine.core.node_count(), seed ^ 1);
    let pre_recall = engine.phasor.report().phase_coherence;
    let sleep = engine.sleep_consolidate().expect("legacy sleep");
    let _ = engine.infer_and_stage(&recall_cues);
    let post_recall = engine.phasor.report().phase_coherence;

    let legacy_handshake = legacy_handshake_metrics(&engine, pairs, tokens);

    ArchitectureTrialMetrics {
        handshake: legacy_handshake,
        efficiency: EfficiencyMetrics {
            wall_time: wake_elapsed,
            minimizer_iterations: wake.minimization.iterations as f32,
            energy_evaluations: wake.minimization.energy_evaluations as f32,
            reservoir_features: 0,
        },
        consolidation: ConsolidationMetrics {
            wake_gate_passed: wake.gate.passed,
            sleep_accepted: sleep.accepted,
            sleep_rejected: sleep.rejected,
            recall_coherence: post_recall,
            recall_alignment: (pre_recall + post_recall) * 0.5,
            thermo_free_energy: wake.minimization.final_report.free_energy,
            ctp_bias_norm: 0.0,
        },
    }
}

fn run_hybrid_trial(
    config: &HybridLegacyComparisonConfig,
    tokens: &[Vec<f32>],
    values: &[Vec<f32>],
    pairs: &[(usize, usize)],
    seed: u64,
) -> ArchitectureTrialMetrics {
    let mut engine = HybridThermoAttention::new(HybridThermoAttentionConfig {
        d_model: config.d_model,
        d_v: config.d_v,
        rff: PhasorRffConfig {
            features: 64,
            seed: seed ^ 0x5246_46,
            ..Default::default()
        },
        cdt_nodes: config.cdt_nodes,
        cdt_seed: seed,
        hybrid: shared_hybrid_config(config),
        ..Default::default()
    })
    .expect("hybrid engine");

    let started = Instant::now();
    let (_, report) = engine.forward(tokens, values).expect("hybrid forward");
    let forward_elapsed = started.elapsed();

    let attention = engine.last_attention().to_vec();
    let handshake = hybrid_handshake_metrics(&engine, &attention, tokens, pairs);
    let sleep = engine.sleep_consolidate().expect("hybrid sleep");

    let recall_tokens = vec![tokens[0].clone()];
    let recall_values = vec![values[0].clone()];
    let _ = engine.forward(&recall_tokens, &recall_values);
    let recall_coherence = engine.hybrid_engine().phasor.report().phase_coherence;

    ArchitectureTrialMetrics {
        handshake,
        efficiency: EfficiencyMetrics {
            wall_time: forward_elapsed,
            minimizer_iterations: report.tick as f32,
            energy_evaluations: 0.0,
            reservoir_features: engine.config().rff.features,
        },
        consolidation: ConsolidationMetrics {
            wake_gate_passed: report.wake_gate_passed,
            sleep_accepted: sleep.accepted,
            sleep_rejected: sleep.rejected,
            recall_coherence,
            recall_alignment: recall_coherence,
            thermo_free_energy: report.thermo_free_energy,
            ctp_bias_norm: report.ctp_bias_norm,
        },
    }
}

fn shared_hybrid_config(_config: &HybridLegacyComparisonConfig) -> NativeHybridConfig {
    NativeHybridConfig {
        minimizer: NativePhasorMinimizerConfig {
            max_iterations: 60,
            residual_tolerance: 1.0e-2,
            handshake_strength: 0.65,
            attention_strength: 0.55,
            inference_policy: crate::native_phasor_thermodynamic_engine::NativePhasorInferencePolicy::Adaptive,
            ..Default::default()
        },
        minimum_relative_energy_drop: 0.0,
        maximum_residual: 1.0e-1,
        minimum_magnetic_coherence: 0.45,
        cue_as_boundary: true,
        ..Default::default()
    }
}

// ── Métricas de apretón de manos ─────────────────────────────────────────────

fn hybrid_handshake_metrics(
    engine: &HybridThermoAttention,
    attention: &[Vec<f32>],
    tokens: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> HandshakeMetrics {
    let (top1, mrr, mass) = planted_pair_metrics(attention, pairs);
    let entropy = attention
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&p| p > EPSILON)
        .map(|&p| -p * p.ln())
        .sum::<f32>()
        / attention.len().max(1) as f32;

    let rff = crate::hybrid_thermo_attention::PhasorRffMap::new(
        engine.config().d_model,
        engine.config().rff,
    );
    let correlation = rff_softmax_correlation(&rff, tokens, attention);

    HandshakeMetrics {
        top1_rate: top1,
        mean_reciprocal_rank: mrr,
        planted_mass: mass,
        softmax_entropy: entropy,
        rff_softmax_correlation: correlation,
    }
}

fn legacy_handshake_metrics(
    engine: &NativeHybridPhasorCdtEngine,
    pairs: &[(usize, usize)],
    tokens: &[Vec<f32>],
) -> HandshakeMetrics {
    let n = tokens.len();
    let nodes = engine.core.node_count();
    let mut hits = 0usize;
    let mut mrr = 0.0f32;
    let mut mass = 0.0f32;

    for &(q, k) in pairs {
        let q_node = token_node(q, nodes, 0);
        let k_node = token_node(k, nodes, 0);
        let q_amp = engine.phasor.phasors[q_node].norm();
        let k_amp = engine.phasor.phasors[k_node].norm();
        let coupling = (q_amp * k_amp).min(1.0);
        mass += coupling;

        let dominant = engine.phasor.dominant_nodes(n);
        let rank = 1 + dominant
            .iter()
            .position(|(node, _, _)| *node == k_node)
            .unwrap_or(n);
        if rank == 1 {
            hits += 1;
        }
        mrr += 1.0 / rank as f32;
    }
    let count = pairs.len().max(1) as f32;
    HandshakeMetrics {
        top1_rate: hits as f32 / count,
        mean_reciprocal_rank: mrr / count,
        planted_mass: mass / count,
        softmax_entropy: 0.0,
        rff_softmax_correlation: 0.0,
    }
}

fn planted_pair_metrics(attention: &[Vec<f32>], pairs: &[(usize, usize)]) -> (f32, f32, f32) {
    if pairs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut top1 = 0usize;
    let mut mrr = 0.0f32;
    let mut mass = 0.0f32;
    for &(q, k) in pairs {
        let row = &attention[q];
        mass += row[k];
        let weight = row[k];
        let rank = 1 + row
            .iter()
            .enumerate()
            .filter(|(j, &a)| *j != k && a > weight)
            .count();
        if rank == 1 {
            top1 += 1;
        }
        mrr += 1.0 / rank as f32;
    }
    let n = pairs.len() as f32;
    (top1 as f32 / n, mrr / n, mass / n)
}

fn rff_softmax_correlation(
    rff: &crate::hybrid_thermo_attention::PhasorRffMap,
    tokens: &[Vec<f32>],
    attention: &[Vec<f32>],
) -> f32 {
    let n = tokens.len();
    if n < 2 {
        return 0.0;
    }
    let mut rff_logits = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            rff_logits[i][j] = rff.kernel_approx(&tokens[i], &tokens[j]);
        }
    }
    pearson_correlation_flat(&flatten(&rff_logits), &flatten(attention))
}

fn pearson_correlation_flat(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt().max(EPSILON);
    (cov / denom).clamp(-1.0, 1.0)
}

fn flatten(matrix: &[Vec<f32>]) -> Vec<f32> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}

// ── Datos sintéticos ─────────────────────────────────────────────────────────

pub fn planted_handshake_sequence(
    n: usize,
    d_model: usize,
    d_v: usize,
    pairs: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<(usize, usize)>) {
    let mut tokens = synthetic_sequence(n, d_model, seed);
    let values = synthetic_sequence(n, d_v, seed ^ 1);
    let mut planted = Vec::new();
    for p in 0..pairs.min(n / 2) {
        let q_idx = p * 2;
        let k_idx = p * 2 + 1;
        tokens[q_idx] = tokens[k_idx].clone();
        planted.push((q_idx, k_idx));
    }
    (tokens, values, planted)
}

fn synthetic_sequence(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..d)
                .map(|j| signed_unit(seed ^ ((i as u64) << 16) ^ j as u64))
                .collect()
        })
        .collect()
}

fn tokens_to_legacy_cues(tokens: &[Vec<f32>], node_count: usize, seed: u64) -> Vec<NativePhasorCue> {
    tokens
        .iter()
        .enumerate()
        .map(|(index, embedding)| {
            let node = token_node(index, node_count, seed);
            let phase = embedding
                .iter()
                .enumerate()
                .map(|(d, &v)| v * (d as f32 + 1.0))
                .sum::<f32>()
                .rem_euclid(std::f32::consts::TAU);
            let amplitude = embedding.iter().map(|v| v * v).sum::<f32>().sqrt() / embedding.len() as f32;
            NativePhasorCue {
                node,
                amplitude: amplitude.max(0.1),
                phase,
            }
        })
        .collect()
}

fn token_node(token_index: usize, node_count: usize, seed: u64) -> usize {
    splitmix64(seed ^ token_index as u64) as usize % node_count.max(1)
}

// ── Tests de regresión ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_thermo_attention::digital_softmax_attention;

    #[test]
    fn comparison_runs_without_panic() {
        let report = run_hybrid_legacy_comparison(HybridLegacyComparisonConfig {
            sequence_length: 8,
            d_model: 8,
            d_v: 4,
            planted_pairs: 2,
            trials: 3,
            cdt_nodes: 64,
            ..Default::default()
        });
        assert_eq!(report.legacy.trials, 3);
        assert_eq!(report.hybrid.trials, 3);
    }

    #[test]
    fn hybrid_handshake_beats_legacy_on_planted_pairs() {
        let report = run_hybrid_legacy_comparison(HybridLegacyComparisonConfig {
            sequence_length: 16,
            d_model: 16,
            d_v: 8,
            planted_pairs: 4,
            trials: 16,
            cdt_nodes: 128,
            ..Default::default()
        });
        assert!(
            report.hybrid.mean_top1() >= report.legacy.mean_top1(),
            "hybrid top1={:.3} legacy top1={:.3}",
            report.hybrid.mean_top1(),
            report.legacy.mean_top1()
        );
        assert!(
            report.hybrid.mean_mrr() >= report.legacy.mean_mrr() * 0.95,
            "hybrid mrr={:.3} legacy mrr={:.3}",
            report.hybrid.mean_mrr(),
            report.legacy.mean_mrr()
        );
    }

    #[test]
    fn hybrid_softmax_ctp_has_positive_rff_correlation() {
        let report = run_hybrid_legacy_comparison(HybridLegacyComparisonConfig {
            sequence_length: 12,
            d_model: 12,
            trials: 8,
            ..Default::default()
        });
        assert!(
            report.hybrid.mean_rff_correlation() > 0.3,
            "correlación RFF-Softmax={:.3}",
            report.hybrid.mean_rff_correlation()
        );
    }

    #[test]
    fn hybrid_consolidation_recall_coherence_matches_legacy() {
        let report = run_hybrid_legacy_comparison(HybridLegacyComparisonConfig {
            trials: 10,
            ..Default::default()
        });
        assert!(
            report.hybrid.mean_recall_coherence() >= report.legacy.mean_recall_coherence() * 0.85,
            "hybrid recall={:.3} legacy recall={:.3}",
            report.hybrid.mean_recall_coherence(),
            report.legacy.mean_recall_coherence()
        );
        assert!(
            report.hybrid.mean_sleep_accepted() >= report.legacy.mean_sleep_accepted() * 0.5,
            "hybrid sleep={:.2} legacy sleep={:.2}",
            report.hybrid.mean_sleep_accepted(),
            report.legacy.mean_sleep_accepted()
        );
    }

    #[test]
    fn hybrid_ctp_bias_is_nonzero_after_forward() {
        let report = run_hybrid_legacy_comparison(HybridLegacyComparisonConfig {
            trials: 5,
            ..Default::default()
        });
        assert!(
            report.hybrid.ctp_bias_norm_sum / report.hybrid.trials.max(1) as f64 > 0.5,
            "B_CTP debe activarse tras relajación termodinámica"
        );
    }

    #[test]
    fn planted_softmax_mass_exceeds_uniform_baseline() {
        let config = HybridLegacyComparisonConfig::default();
        let (tokens, values, pairs) = planted_handshake_sequence(
            config.sequence_length,
            config.d_model,
            config.d_v,
            config.planted_pairs,
            config.seed,
        );
        let (_, attention) = digital_softmax_attention(&tokens, &tokens, &values, None, 0.0);
        let (_, _, mass) = planted_pair_metrics(&attention, &pairs);
        let uniform = 1.0 / config.sequence_length as f32;
        assert!(mass > uniform, "mass={mass} uniform={uniform}");
    }

    #[test]
    fn multi_seed_handshake_regression() {
        let mut hybrid_wins = 0usize;
        let seeds = [0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6];
        for &seed in &seeds {
            let report = run_hybrid_legacy_comparison(HybridLegacyComparisonConfig {
                seed,
                trials: 8,
                ..Default::default()
            });
            if report.hybrid.mean_mrr() >= report.legacy.mean_mrr() {
                hybrid_wins += 1;
            }
        }
        assert!(
            hybrid_wins >= 4,
            "hybrid debe ganar o empatar MRR en al menos 4/6 semillas"
        );
    }
}
