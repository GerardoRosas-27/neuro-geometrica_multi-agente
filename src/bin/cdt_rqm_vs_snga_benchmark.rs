use snga::cdt_graphity::{CdtGraphityConfig, CdtGraphitySubstrate};
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

const EPOCHS: usize = 8;
const NODES_PER_SLICE: usize = 24;

#[derive(Clone, Copy)]
struct Frame {
    observer: ObserverId,
    phase: f32,
    context: usize,
    effect: usize,
}

#[derive(Clone, Copy)]
struct AmbiguousCausalCase {
    cue: usize,
    left: Frame,
    right: Frame,
}

#[derive(Default)]
struct Metrics {
    frames: usize,
    top_correct: usize,
    purity_sum: f32,
    leakage_sum: f32,
    margin_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32) {
        let total = expected + distractor;
        let purity = if total > f32::EPSILON {
            expected / total
        } else {
            0.0
        };
        let leakage = if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.frames += 1;
        self.top_correct += usize::from(expected > distractor);
        self.purity_sum += purity;
        self.leakage_sum += leakage;
        self.margin_sum += expected - distractor;
    }

    fn accuracy(&self) -> f32 {
        self.top_correct as f32 / self.frames.max(1) as f32
    }

    fn purity(&self) -> f32 {
        self.purity_sum / self.frames.max(1) as f32
    }

    fn leakage(&self) -> f32 {
        self.leakage_sum / self.frames.max(1) as f32
    }

    fn margin(&self) -> f32 {
        self.margin_sum / self.frames.max(1) as f32
    }
}

fn main() {
    let cases = cases();
    let mut snga = SimplicialNetwork::grid_3d(snga_config(), 2);
    let mut cdt_rqm = CdtRqmUniverseSubstrate::new(cdt_rqm_config());
    let mut cdt_only = CdtGraphitySubstrate::graphity_hot_start(cdt_config());

    let mut early_snga_cue = Metrics::default();
    let mut early_snga_context = Metrics::default();
    let mut early_cdt_rqm = Metrics::default();
    let mut early_cdt_only_score = 0.0;

    for epoch in 0..EPOCHS {
        train_epoch(&mut snga, &mut cdt_rqm, &mut cdt_only, &cases);
        if epoch == 1 {
            early_snga_cue = evaluate_snga(&snga, &cases, QueryMode::CueOnly);
            early_snga_context = evaluate_snga(&snga, &cases, QueryMode::CueAndContext);
            early_cdt_rqm = evaluate_cdt_rqm(&cdt_rqm, &cases);
            early_cdt_only_score = evaluate_cdt_only(&cdt_only, &cases);
        }
    }

    let final_snga_cue = evaluate_snga(&snga, &cases, QueryMode::CueOnly);
    let final_snga_context = evaluate_snga(&snga, &cases, QueryMode::CueAndContext);
    let final_cdt_rqm = evaluate_cdt_rqm(&cdt_rqm, &cases);
    let final_cdt_only_score = evaluate_cdt_only(&cdt_only, &cases);
    let final_hardware = cdt_rqm.hardware.step(&[]);

    println!("CDT-RQM vs previous SNGA benchmark");
    println!(
        "cases={} frames={} epochs={} cdt_rqm_relations={}",
        cases.len(),
        cases.len() * 2,
        EPOCHS,
        cdt_rqm.relation_count()
    );
    print_metrics("early_snga_cue_only", &early_snga_cue);
    print_metrics("early_snga_context", &early_snga_context);
    println!("early_cdt_only_score={:.1}%", early_cdt_only_score * 100.0);
    print_metrics("early_cdt_rqm", &early_cdt_rqm);
    print_metrics("final_snga_cue_only", &final_snga_cue);
    print_metrics("final_snga_context", &final_snga_context);
    println!("final_cdt_only_score={:.1}%", final_cdt_only_score * 100.0);
    print_metrics("final_cdt_rqm", &final_cdt_rqm);
    println!(
        "cdt_rqm_hardware: temp={:.3} active_edges={} regge={:.3} causality_violations={}",
        final_hardware.temperature,
        final_hardware.active_edges,
        final_hardware.regge_action,
        final_hardware.causality_violations
    );
    println!(
        "lectura: {}",
        if final_cdt_rqm.accuracy() >= final_snga_context.accuracy()
            && final_cdt_rqm.leakage() < final_snga_context.leakage()
            && final_hardware.causality_violations == 0
        {
            "CDT-RQM separa futuros relativos por observador y reduce fuga frente al SNGA causal anterior"
        } else {
            "CDT-RQM aprende, pero aun requiere calibracion para superar claramente al SNGA anterior"
        }
    );
}

fn train_epoch(
    snga: &mut SimplicialNetwork,
    cdt_rqm: &mut CdtRqmUniverseSubstrate,
    cdt_only: &mut CdtGraphitySubstrate,
    cases: &[AmbiguousCausalCase],
) {
    for case in cases {
        train_frame(snga, cdt_rqm, cdt_only, case.cue, case.left);
        train_frame(snga, cdt_rqm, cdt_only, case.cue, case.right);
    }
}

fn train_frame(
    snga: &mut SimplicialNetwork,
    cdt_rqm: &mut CdtRqmUniverseSubstrate,
    cdt_only: &mut CdtGraphitySubstrate,
    cue: usize,
    frame: Frame,
) {
    let cue_pattern = cue_pattern(cue);
    let context_pattern = context_pattern(frame.context);
    let effect_pattern = effect_pattern(frame.effect);
    let mut fused = cue_pattern.clone();
    fused.extend(context_pattern.iter().copied());
    fused.extend(effect_pattern.iter().copied());
    fused.sort_unstable();
    fused.dedup();

    snga.learn_transition(&cue_pattern, &effect_pattern);
    snga.learn_transition(&context_pattern, &effect_pattern);
    snga.reinforce_coactivation_if_useful(&fused, 0.045, 0.92);

    cdt_only.clear_activity();
    cdt_only.inject_pattern(&cue_pattern, 1.0);
    cdt_only.step(&effect_pattern);

    cdt_rqm.hardware.clear_activity();
    cdt_rqm.train_observed_transition(
        frame.observer,
        frame.phase,
        &cue_pattern,
        &effect_pattern,
        1.0,
    );
}

#[derive(Clone, Copy)]
enum QueryMode {
    CueOnly,
    CueAndContext,
}

fn evaluate_snga(
    network: &SimplicialNetwork,
    cases: &[AmbiguousCausalCase],
    mode: QueryMode,
) -> Metrics {
    let mut metrics = Metrics::default();
    for case in cases {
        evaluate_snga_frame(network, case.cue, case.left, case.right, mode, &mut metrics);
        evaluate_snga_frame(network, case.cue, case.right, case.left, mode, &mut metrics);
    }
    metrics
}

fn evaluate_snga_frame(
    network: &SimplicialNetwork,
    cue: usize,
    frame: Frame,
    competing: Frame,
    mode: QueryMode,
    metrics: &mut Metrics,
) {
    let mut query = cue_pattern(cue);
    if matches!(mode, QueryMode::CueAndContext) {
        query.extend(context_pattern(frame.context));
        query.sort_unstable();
        query.dedup();
    }
    let prediction = network.predict_from(&query, 96);
    let expected = score_prediction(&prediction, &effect_pattern(frame.effect));
    let distractor = score_prediction(&prediction, &effect_pattern(competing.effect));
    metrics.record(expected, distractor);
}

fn evaluate_cdt_rqm(substrate: &CdtRqmUniverseSubstrate, cases: &[AmbiguousCausalCase]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for case in cases {
        evaluate_cdt_rqm_frame(&mut trial, case.cue, case.left, case.right, &mut metrics);
        evaluate_cdt_rqm_frame(&mut trial, case.cue, case.right, case.left, &mut metrics);
    }
    metrics
}

fn evaluate_cdt_rqm_frame(
    substrate: &mut CdtRqmUniverseSubstrate,
    cue: usize,
    frame: Frame,
    competing: Frame,
    metrics: &mut Metrics,
) {
    let cue_pattern = cue_pattern(cue);
    substrate.hardware.clear_activity();
    substrate.hardware.inject_pattern(&cue_pattern, 1.0);
    let report = substrate.step_from_boundary(frame.observer, frame.phase, &cue_pattern);
    let expected = score_collapse(&report.collapse, &effect_pattern(frame.effect));
    let distractor = score_collapse(&report.collapse, &effect_pattern(competing.effect));
    metrics.record(expected, distractor);
}

fn evaluate_cdt_only(substrate: &CdtGraphitySubstrate, cases: &[AmbiguousCausalCase]) -> f32 {
    let mut trial = substrate.clone();
    let mut matched = 0;
    let mut total = 0;
    for case in cases {
        for frame in [case.left, case.right] {
            let cue_pattern = cue_pattern(case.cue);
            trial.clear_activity();
            trial.inject_pattern(&cue_pattern, 1.0);
            let predicted = trial.predict_next(&cue_pattern, 12);
            for expected in effect_pattern(frame.effect) {
                total += 1;
                if predicted.iter().any(|(idx, _)| *idx == expected) {
                    matched += 1;
                }
            }
        }
    }
    matched as f32 / total.max(1) as f32
}

fn score_prediction(prediction: &[(usize, f32)], targets: &[usize]) -> f32 {
    prediction
        .iter()
        .filter(|(idx, _)| targets.contains(idx))
        .map(|(_, score)| *score)
        .sum()
}

fn score_collapse(report: &snga::relational_field::CollapseReport, targets: &[usize]) -> f32 {
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn print_metrics(label: &str, metrics: &Metrics) {
    println!(
        "{}: accuracy={:.1}% purity={:.1}% leakage={:.1}% margin={:.3}",
        label,
        metrics.accuracy() * 100.0,
        metrics.purity() * 100.0,
        metrics.leakage() * 100.0,
        metrics.margin()
    );
}

fn cue_pattern(ordinal: usize) -> Vec<usize> {
    pattern(0, ordinal)
}

fn effect_pattern(ordinal: usize) -> Vec<usize> {
    pattern(1, ordinal)
}

fn context_pattern(ordinal: usize) -> Vec<usize> {
    pattern(2, ordinal)
}

fn pattern(slice: usize, ordinal: usize) -> Vec<usize> {
    let base = slice * NODES_PER_SLICE + ordinal;
    vec![base, base + 1, base + 2]
}

fn cdt_rqm_config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: cdt_config(),
        rqm: rqm_config(),
        max_quantum_candidates: 12,
        rqm_feedback_gain: 0.40,
    }
}

fn rqm_config() -> RelationalFieldConfig {
    RelationalFieldConfig {
        amplitude_learning_rate: 0.09,
        phase_learning_rate: 0.22,
        coherence_learning_rate: 0.12,
        uncertainty_learning_rate: 0.10,
        amplitude_decay: 0.001,
        coherence_decay: 0.0005,
        uncertainty_recovery: 0.002,
        activation_threshold: 0.025,
    }
}

fn cdt_config() -> CdtGraphityConfig {
    CdtGraphityConfig {
        slices: 5,
        nodes_per_slice: NODES_PER_SLICE,
        initial_spatial_connectivity: 0.34,
        initial_temporal_connectivity: 0.18,
        target_spatial_degree: 5,
        target_temporal_degree: 3,
        target_tetrahedra_per_edge: 4,
        cooling_rate: 0.055,
        heating_rate: 0.12,
        reinforcement_rate: 0.11,
        prune_threshold: 0.055,
        max_new_edges_per_step: 8,
        seed: 9_144,
    }
}

fn snga_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 32,
        height: 12,
        spacing: 8.0,
        elasticity: 0.006,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 40,
        inhibition_decay: 0.05,
        max_spikes_per_step: 96,
        local_inhibition_decay: 0.70,
        refractory_ticks: 1,
        rhythm_period: 16,
        rhythm_amplitude: 0.0,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 128,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.075,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 9_145,
    }
}

fn cases() -> Vec<AmbiguousCausalCase> {
    vec![
        ambiguous(
            0,
            frame(1, 0.0, 0, 1),
            frame(2, std::f32::consts::FRAC_PI_2, 1, 5),
        ),
        ambiguous(
            4,
            frame(1, 0.0, 2, 9),
            frame(2, std::f32::consts::FRAC_PI_2, 3, 13),
        ),
        ambiguous(
            8,
            frame(1, 0.0, 4, 17),
            frame(2, std::f32::consts::FRAC_PI_2, 5, 19),
        ),
        ambiguous(
            10,
            frame(1, 0.0, 6, 3),
            frame(2, std::f32::consts::FRAC_PI_2, 7, 15),
        ),
    ]
}

fn ambiguous(cue: usize, left: Frame, right: Frame) -> AmbiguousCausalCase {
    AmbiguousCausalCase { cue, left, right }
}

fn frame(observer: usize, phase: f32, context: usize, effect: usize) -> Frame {
    Frame {
        observer: ObserverId(observer),
        phase,
        context,
        effect,
    }
}
