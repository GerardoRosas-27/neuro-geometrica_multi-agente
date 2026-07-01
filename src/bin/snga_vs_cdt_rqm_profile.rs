use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

const SNGA_EPOCHS: usize = 16;
const NODES_PER_SLICE: usize = 64;
const ANNEAL_ATTEMPTS: usize = 64;

#[derive(Clone, Copy)]
struct Lesson {
    observer: ObserverId,
    phase: f32,
    cue: usize,
    effect: usize,
    competing_effect: usize,
}

#[derive(Default, Clone, Copy)]
struct MemoryMetrics {
    cases: usize,
    correct: usize,
    purity_sum: f32,
    leakage_sum: f32,
    margin_sum: f32,
}

impl MemoryMetrics {
    fn record(&mut self, expected: f32, distractor: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.purity_sum += if total > f32::EPSILON {
            expected / total
        } else {
            0.0
        };
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

    fn purity(self) -> f32 {
        self.purity_sum / self.cases.max(1) as f32
    }

    fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    fn margin(self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }
}

fn main() {
    let lessons = lessons();
    let mut snga = SimplicialNetwork::grid_3d(snga_config(), 2);
    train_snga(&mut snga, &lessons);
    let snga_metrics = evaluate_snga(&snga, &lessons);
    let snga_profile = profile_snga(&mut snga.clone(), &lessons);

    let mut cdt_rqm = CdtRqmUniverseSubstrate::new(cdt_rqm_config());
    cdt_rqm.migrate_snga_causal_edges(ObserverId(900), 0.0, snga.causal_edges_snapshot(), 0.05);
    distill_snga_predictions(&snga, &mut cdt_rqm, &lessons);
    let before_edges = active_cdt_edges(&cdt_rqm);
    let before_regge = cdt_rqm.hardware.regge_action();
    let validation = validation_set(&lessons);
    let anneal = cdt_rqm.anneal_after_migration(&validation, ANNEAL_ATTEMPTS);
    let cdt_metrics = evaluate_cdt_rqm(&cdt_rqm, &lessons);
    let cdt_profile = profile_cdt_rqm(&mut cdt_rqm.clone(), &lessons);

    println!("SNGA vs CDT-RQM trained substrate profile");
    println!(
        "lessons={} snga_epochs={} anneal_attempts={}",
        lessons.len(),
        SNGA_EPOCHS,
        ANNEAL_ATTEMPTS
    );
    print_memory("snga_memory", snga_metrics);
    print_memory("cdt_rqm_memory", cdt_metrics);
    println!(
        "snga_structure: total_nodes={} active_nodes_avg={:.1} total_edges={} active_edges={} associative_edges={} causal_edges={} semantic_cells={} free_energy_avg={:.3} knowledge_mass={:.3}",
        snga.agents.len(),
        snga_profile.active_nodes_avg,
        snga.edges.len(),
        snga.plasticity_stats().active_edges,
        snga.plasticity_stats().associative_edges,
        snga.plasticity_stats().causal_edges,
        snga.plasticity_stats().semantic_cells,
        snga_profile.energy_avg,
        snga_profile.knowledge_mass
    );
    println!(
        "cdt_rqm_structure: total_nodes={} active_nodes_avg={:.1} total_edges={} active_edges={} spatial_edges={} temporal_edges={} rqm_relations={} regge={:.3} temperature={:.3} knowledge_mass={:.3}",
        cdt_rqm.hardware.nodes.len(),
        cdt_profile.active_nodes_avg,
        cdt_rqm.hardware.edges.len(),
        cdt_profile.active_edges,
        cdt_profile.spatial_edges,
        cdt_profile.temporal_edges,
        cdt_rqm.relation_count(),
        cdt_profile.regge,
        cdt_rqm.hardware.temperature,
        cdt_profile.knowledge_mass
    );
    println!(
        "optimization: cdt_edges_before={} cdt_edges_after={} edge_compression={:.1}% regge_before={:.3} regge_after={:.3} regge_reduction={:.1}% anneal_accepted={}",
        before_edges,
        cdt_profile.active_edges,
        (1.0 - cdt_profile.active_edges as f32 / before_edges.max(1) as f32) * 100.0,
        before_regge,
        cdt_profile.regge,
        (1.0 - cdt_profile.regge / before_regge.max(1.0)) * 100.0,
        anneal.accepted
    );
    println!(
        "efficiency: snga_accuracy_per_active_edge={:.6} cdt_accuracy_per_active_edge={:.6} snga_leakage={:.1}% cdt_leakage={:.1}% causality_violations={}",
        snga_metrics.accuracy() / snga.plasticity_stats().active_edges.max(1) as f32,
        cdt_metrics.accuracy() / cdt_profile.active_edges.max(1) as f32,
        snga_metrics.leakage() * 100.0,
        cdt_metrics.leakage() * 100.0,
        cdt_rqm.hardware.causality_violations()
    );
    println!(
        "lectura: {}",
        if cdt_metrics.accuracy() >= snga_metrics.accuracy()
            && cdt_metrics.leakage() <= snga_metrics.leakage()
            && cdt_profile.active_edges < snga.plasticity_stats().active_edges
        {
            "CDT-RQM conserva conocimiento comparable con menor fuga y una geometria activa mas compacta"
        } else {
            "CDT-RQM conserva conocimiento, pero aun no supera todos los indicadores estructurales de SNGA"
        }
    );
}

struct SngaProfile {
    active_nodes_avg: f32,
    energy_avg: f32,
    knowledge_mass: f32,
}

struct CdtProfile {
    active_nodes_avg: f32,
    active_edges: usize,
    spatial_edges: usize,
    temporal_edges: usize,
    regge: f32,
    knowledge_mass: f32,
}

fn profile_snga(network: &mut SimplicialNetwork, lessons: &[Lesson]) -> SngaProfile {
    let mut active_sum = 0_usize;
    let mut energy_sum = 0.0;
    for lesson in lessons {
        network.clear_activity();
        network.inject_pattern(&cue_pattern(lesson.cue), 1.0, 3);
        for _ in 0..3 {
            network.step();
        }
        active_sum += network
            .agents
            .iter()
            .filter(|agent| agent.surprise > 0.05)
            .count();
        energy_sum += network.total_free_energy();
    }
    let knowledge_mass = network
        .causal_edges_snapshot()
        .iter()
        .map(|(_, _, weight)| *weight)
        .sum::<f32>()
        + network
            .edges
            .iter()
            .filter(|edge| edge.active)
            .map(|edge| edge.weight)
            .sum::<f32>();
    SngaProfile {
        active_nodes_avg: active_sum as f32 / lessons.len().max(1) as f32,
        energy_avg: energy_sum / lessons.len().max(1) as f32,
        knowledge_mass,
    }
}

fn profile_cdt_rqm(substrate: &mut CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> CdtProfile {
    let mut active_sum = 0_usize;
    for lesson in lessons {
        let cue = cue_pattern(lesson.cue);
        substrate.hardware.clear_activity();
        substrate.hardware.inject_pattern(&cue, 1.0);
        substrate.step_from_boundary(lesson.observer, lesson.phase, &cue);
        active_sum += substrate
            .hardware
            .nodes
            .iter()
            .filter(|node| node.surprise > 0.05)
            .count();
    }
    let active_edges = active_cdt_edges(substrate);
    let spatial_edges = substrate
        .hardware
        .edges
        .iter()
        .filter(|edge| {
            edge.active && matches!(edge.kind, snga::cdt_graphity::CdtGraphityEdgeKind::Spatial)
        })
        .count();
    let temporal_edges = substrate
        .hardware
        .edges
        .iter()
        .filter(|edge| {
            edge.active && matches!(edge.kind, snga::cdt_graphity::CdtGraphityEdgeKind::Temporal)
        })
        .count();
    let knowledge_mass = substrate
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .map(|edge| edge.stability)
        .sum::<f32>()
        + substrate.relation_count() as f32;
    CdtProfile {
        active_nodes_avg: active_sum as f32 / lessons.len().max(1) as f32,
        active_edges,
        spatial_edges,
        temporal_edges,
        regge: substrate.hardware.regge_action(),
        knowledge_mass,
    }
}

fn active_cdt_edges(substrate: &CdtRqmUniverseSubstrate) -> usize {
    substrate
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .count()
}

fn train_snga(network: &mut SimplicialNetwork, lessons: &[Lesson]) {
    for _ in 0..SNGA_EPOCHS {
        for lesson in lessons {
            let cue = cue_pattern(lesson.cue);
            let effect = effect_pattern(lesson.effect);
            let mut fused = cue.clone();
            fused.extend(effect.iter().copied());
            fused.sort_unstable();
            fused.dedup();
            network.learn_transition(&cue, &effect);
            network.reinforce_coactivation_if_useful(&fused, 0.04, 0.92);
        }
    }
}

fn distill_snga_predictions(
    snga: &SimplicialNetwork,
    substrate: &mut CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
) {
    for lesson in lessons {
        let cue = cue_pattern(lesson.cue);
        for (target, score) in snga.predict_from(&cue, 24) {
            if score <= 0.0 || !effect_pattern(lesson.effect).contains(&target) {
                continue;
            }
            for &source in &cue {
                for _ in 0..4 {
                    substrate.software.reinforce_relation(
                        lesson.observer,
                        source,
                        target,
                        lesson.phase,
                        score.min(1.0),
                    );
                }
                substrate
                    .hardware
                    .reinforce_temporal_link(source, target, score.min(1.0));
            }
        }
    }
}

fn evaluate_snga(network: &SimplicialNetwork, lessons: &[Lesson]) -> MemoryMetrics {
    let mut metrics = MemoryMetrics::default();
    for lesson in lessons {
        let prediction = network.predict_from(&cue_pattern(lesson.cue), 96);
        metrics.record(
            score_prediction(&prediction, &effect_pattern(lesson.effect)),
            score_prediction(&prediction, &effect_pattern(lesson.competing_effect)),
        );
    }
    metrics
}

fn evaluate_cdt_rqm(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> MemoryMetrics {
    let mut trial = substrate.clone();
    let mut metrics = MemoryMetrics::default();
    for lesson in lessons {
        let cue = cue_pattern(lesson.cue);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let report = trial.step_from_boundary(lesson.observer, lesson.phase, &cue);
        metrics.record(
            score_collapse(&report.collapse, &effect_pattern(lesson.effect)),
            score_collapse(&report.collapse, &effect_pattern(lesson.competing_effect)),
        );
    }
    metrics
}

fn validation_set(
    lessons: &[Lesson],
) -> Vec<(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)> {
    lessons
        .iter()
        .map(|lesson| {
            (
                lesson.observer,
                lesson.phase,
                cue_pattern(lesson.cue),
                effect_pattern(lesson.effect),
                effect_pattern(lesson.competing_effect),
            )
        })
        .collect()
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

fn print_memory(label: &str, metrics: MemoryMetrics) {
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

fn pattern(slice: usize, ordinal: usize) -> Vec<usize> {
    let base = slice * NODES_PER_SLICE + ordinal;
    vec![base, base + 1, base + 2]
}

fn lessons() -> Vec<Lesson> {
    let mut out = Vec::new();
    let phases = [
        0.0,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        -std::f32::consts::FRAC_PI_2,
    ];
    for group in 0..4 {
        let observer = ObserverId(group + 1);
        let phase = phases[group];
        for offset in 0..5 {
            let cue = group * 12 + offset * 2;
            let effect = group * 12 + offset * 2 + 1;
            let competing = group * 12 + ((offset + 2) % 5) * 2 + 1;
            out.push(Lesson {
                observer,
                phase,
                cue,
                effect,
                competing_effect: competing,
            });
        }
    }
    out
}

fn cdt_rqm_config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: cdt_config(),
        rqm: rqm_config(),
        max_quantum_candidates: 16,
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
        slices: 4,
        nodes_per_slice: NODES_PER_SLICE,
        initial_spatial_connectivity: 0.22,
        initial_temporal_connectivity: 0.10,
        target_spatial_degree: 5,
        target_temporal_degree: 3,
        target_tetrahedra_per_edge: 4,
        cooling_rate: 0.055,
        heating_rate: 0.12,
        reinforcement_rate: 0.11,
        prune_threshold: 0.055,
        max_new_edges_per_step: 8,
        seed: 15_144,
    }
}

fn snga_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 32,
        height: 16,
        spacing: 8.0,
        elasticity: 0.006,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 64,
        inhibition_decay: 0.05,
        max_spikes_per_step: 128,
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
        seed: 15_145,
    }
}
