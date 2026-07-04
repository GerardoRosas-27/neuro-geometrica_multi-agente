use cdt_rqm_epr::cdt_graphity::CdtGraphityConfig;
use cdt_rqm_epr::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::relational_field::{CandidateScore, CollapseReport, ObserverId, RelationalFieldConfig};

const NODES_PER_SLICE: usize = 128;

#[derive(Clone, Copy)]
struct Lesson {
    observer: ObserverId,
    phase: f32,
    cue: usize,
    target: usize,
    distractor: usize,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    cases: usize,
    correct: usize,
    leakage_sum: f32,
    margin_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
    }

    fn accuracy(self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    fn margin(self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }
}

#[derive(Clone, Copy, Default)]
struct PointerStats {
    observed: usize,
    selected: usize,
    mean_pointer_score: f32,
}

fn main() {
    let lessons = lessons();
    let contextual = contextual_lessons();

    let mut baseline = CdtRqmUniverseSubstrate::new(config());
    baseline.enable_entanglement(epr_config());
    train(&mut baseline, &lessons, 8);
    let baseline_metrics = evaluate(&baseline, &lessons, EvaluationMode::Deterministic);
    let born_metrics = evaluate(&baseline, &lessons, EvaluationMode::BornSampled);
    let baseline_pointer_stats = pointer_stats(&baseline, &lessons);

    let mut contextual_substrate = CdtRqmUniverseSubstrate::new(config());
    contextual_substrate.enable_entanglement(epr_config());
    train(&mut contextual_substrate, &contextual, 8);
    let contextual_metrics = evaluate(
        &contextual_substrate,
        &contextual,
        EvaluationMode::Deterministic,
    );
    let contextual_cross_leakage = cross_observer_leakage(&contextual_substrate, &contextual);

    let mut uncertainty_limited = CdtRqmUniverseSubstrate::new(uncertainty_limited_config());
    uncertainty_limited.enable_entanglement(epr_config());
    train(&mut uncertainty_limited, &lessons, 8);
    let uncertainty_metrics = evaluate(
        &uncertainty_limited,
        &lessons,
        EvaluationMode::Deterministic,
    );
    let uncertainty_pointer_stats = pointer_stats(&uncertainty_limited, &lessons);

    println!("CDT-RQM relational quantum measurement validation");
    print_metrics("deterministic_rqm", baseline_metrics);
    print_metrics("born_sampled_collapse", born_metrics);
    print_metrics("kochen_specker_contextuality", contextual_metrics);
    println!(
        "contextual_cross_observer_leakage={:.1}%",
        contextual_cross_leakage * 100.0
    );
    print_metrics("amplitude_phase_uncertainty_budget", uncertainty_metrics);
    println!(
        "einselection: baseline_selected={}/{} mean_pointer={:.3} uncertainty_selected={}/{} mean_pointer={:.3}",
        baseline_pointer_stats.selected,
        baseline_pointer_stats.observed,
        baseline_pointer_stats.mean_pointer_score,
        uncertainty_pointer_stats.selected,
        uncertainty_pointer_stats.observed,
        uncertainty_pointer_stats.mean_pointer_score
    );

    println!(
        "decision_born_rule: {}",
        if born_metrics.accuracy() >= baseline_metrics.accuracy()
            && born_metrics.leakage() <= baseline_metrics.leakage()
            && born_metrics.margin() > baseline_metrics.margin()
        {
            "keep value=improves_sampling_without_memory_loss"
        } else {
            "discard value=stochastic_sampling_reduces_or_does_not_improve_recall"
        }
    );
    println!(
        "decision_contextuality: {}",
        if contextual_metrics.accuracy() >= 1.0 && contextual_cross_leakage < 0.05 {
            "keep_existing value=observer_context_already_separates_incompatible_measurements"
        } else {
            "needs_work value=contextual_observers_interfere"
        }
    );
    println!(
        "decision_decoherence_einselection: {}",
        if baseline_pointer_stats.mean_pointer_score >= 0.80 {
            "keep_as_metric value=stable_pointer_states_are_measurable"
        } else {
            "discard value=pointer_state_signal_too_weak"
        }
    );
    println!(
        "decision_amplitude_phase_uncertainty: {}",
        if uncertainty_metrics.accuracy() >= baseline_metrics.accuracy()
            && uncertainty_metrics.leakage() <= baseline_metrics.leakage()
            && uncertainty_pointer_stats.mean_pointer_score
                > baseline_pointer_stats.mean_pointer_score
        {
            "keep value=improves_generalization_budget"
        } else {
            "discard value=no_gain_over_current_rqm_learning"
        }
    );
}

#[derive(Clone, Copy)]
enum EvaluationMode {
    Deterministic,
    BornSampled,
}

fn train(substrate: &mut CdtRqmUniverseSubstrate, lessons: &[Lesson], epochs: usize) {
    for _ in 0..epochs {
        for lesson in lessons {
            let cue = pattern(0, lesson.cue);
            let target = pattern(1, lesson.target);
            substrate.hardware.clear_activity();
            substrate.train_observed_transition(lesson.observer, lesson.phase, &cue, &target, 1.0);
            for (&a, &b) in cue.iter().zip(target.iter()) {
                substrate.observe_entanglement_correlation(a, b, 0.40);
            }
        }
    }
}

fn evaluate(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    mode: EvaluationMode,
) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let cue = pattern(0, lesson.cue);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let mut report = trial.step_from_boundary(lesson.observer, lesson.phase, &cue);
        if matches!(mode, EvaluationMode::BornSampled) {
            report.collapse.candidates = born_sample_candidates(&report.collapse, lesson.cue);
        }
        let expected = score(&report.collapse, &pattern(1, lesson.target));
        let distractor = score(&report.collapse, &pattern(1, lesson.distractor));
        metrics.record(expected, distractor);
    }
    metrics
}

fn born_sample_candidates(report: &CollapseReport, salt: usize) -> Vec<CandidateScore> {
    if report.candidates.is_empty() {
        return Vec::new();
    }
    let total = report
        .candidates
        .iter()
        .map(|candidate| candidate.score.max(0.0).powi(2))
        .sum::<f32>();
    if total <= f32::EPSILON {
        return vec![report.candidates[0].clone()];
    }
    let mut draw = deterministic_unit(report.observer.0, salt) * total;
    for candidate in &report.candidates {
        draw -= candidate.score.max(0.0).powi(2);
        if draw <= 0.0 {
            return vec![candidate.clone()];
        }
    }
    vec![report.candidates[0].clone()]
}

fn pointer_stats(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> PointerStats {
    let mut observed = 0;
    let mut selected = 0;
    let mut pointer_sum = 0.0;
    for lesson in lessons {
        let cue = pattern(0, lesson.cue);
        let target = pattern(1, lesson.target);
        for &source in &cue {
            for &target in &target {
                if let Some(state) =
                    substrate
                        .software
                        .relation_state(lesson.observer, source, target)
                {
                    let pointer_score =
                        state.amplitude * state.coherence * (1.0 - state.uncertainty);
                    observed += 1;
                    selected += usize::from(
                        state.coherence >= 0.80
                            && state.uncertainty <= 0.20
                            && state.amplitude >= 0.40,
                    );
                    pointer_sum += pointer_score;
                }
            }
        }
    }
    PointerStats {
        observed,
        selected,
        mean_pointer_score: pointer_sum / observed.max(1) as f32,
    }
}

fn cross_observer_leakage(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> f32 {
    let mut trial = substrate.clone();
    let mut leakage_sum = 0.0;
    let mut cases = 0;
    for lesson in lessons {
        let cue = pattern(0, lesson.cue);
        for other in lessons {
            if other.observer == lesson.observer {
                continue;
            }
            trial.hardware.clear_activity();
            trial.hardware.inject_pattern(&cue, 1.0);
            let report = trial.step_from_boundary(other.observer, other.phase, &cue);
            let own = score(&report.collapse, &pattern(1, lesson.target));
            let other_score = score(&report.collapse, &pattern(1, other.target));
            leakage_sum += own / (own + other_score).max(0.0001);
            cases += 1;
        }
    }
    leakage_sum / cases.max(1) as f32
}

fn score(report: &CollapseReport, targets: &[usize]) -> f32 {
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn print_metrics(label: &str, metrics: Metrics) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3}",
        label,
        metrics.accuracy() * 100.0,
        metrics.leakage() * 100.0,
        metrics.margin()
    );
}

fn deterministic_unit(observer: usize, salt: usize) -> f32 {
    let mut value = (observer as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt as u64)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 30;
    ((value & 0xFFFF_FFFF) as f32) / (u32::MAX as f32)
}

fn pattern(slice: usize, ordinal: usize) -> Vec<usize> {
    let base = slice * NODES_PER_SLICE + ordinal * 3;
    vec![base, base + 1, base + 2]
}

fn lessons() -> Vec<Lesson> {
    let phases = [
        0.0,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        -std::f32::consts::FRAC_PI_2,
    ];
    let mut out = Vec::new();
    for group in 0..4 {
        for offset in 0..8 {
            out.push(Lesson {
                observer: ObserverId(140_000 + group),
                phase: phases[group],
                cue: group * 20 + offset * 2,
                target: group * 20 + offset * 2 + 1,
                distractor: group * 20 + ((offset + 3) % 8) * 2 + 1,
            });
        }
    }
    out
}

fn contextual_lessons() -> Vec<Lesson> {
    let phases = [
        0.0,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        -std::f32::consts::FRAC_PI_2,
    ];
    let mut out = Vec::new();
    for observer in 0..4 {
        for offset in 0..8 {
            out.push(Lesson {
                observer: ObserverId(150_000 + observer),
                phase: phases[observer],
                cue: offset,
                target: observer * 20 + offset * 2 + 1,
                distractor: ((observer + 1) % 4) * 20 + offset * 2 + 1,
            });
        }
    }
    out
}

fn config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice: NODES_PER_SLICE,
            initial_spatial_connectivity: 0.20,
            initial_temporal_connectivity: 0.08,
            target_spatial_degree: 5,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 8,
            seed: 57_991,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.09,
            phase_learning_rate: 0.22,
            coherence_learning_rate: 0.12,
            uncertainty_learning_rate: 0.10,
            amplitude_decay: 0.001,
            coherence_decay: 0.0005,
            uncertainty_recovery: 0.002,
            activation_threshold: 0.025,
        },
        max_quantum_candidates: 24,
        rqm_feedback_gain: 0.40,
    }
}

fn uncertainty_limited_config() -> CdtRqmConfig {
    let mut config = config();
    config.rqm.phase_learning_rate = 0.12;
    config.rqm.uncertainty_learning_rate = 0.16;
    config.rqm.coherence_learning_rate = 0.08;
    config
}

fn epr_config() -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 256,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    }
}
