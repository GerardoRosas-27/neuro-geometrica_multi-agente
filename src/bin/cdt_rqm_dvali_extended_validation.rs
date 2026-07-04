use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::cdt_rqm_experimental::{
    dvali_extended_report, landauer_report, unified_free_energy, DvaliExtendedReport,
    UnifiedFreeEnergyConfig,
};
use snga::entanglement::EntanglementConfig;
use snga::relational_field::{CollapseReport, ObserverId, RelationalFieldConfig};
use std::hash::{Hash, Hasher};
use std::path::Path;

const NODES_PER_SLICE: usize = 160;
const OBSERVER: ObserverId = ObserverId(260_001);
const KEPT_STATE: &str = "data/cdt_rqm_epr_dvali_extended_kept.cdt_rqm";

#[derive(Clone)]
struct Lesson {
    local: Vec<usize>,
    remote: Vec<usize>,
    distractor: Vec<usize>,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    cases: usize,
    correct: usize,
    leakage_sum: f32,
    margin_sum: f32,
    prediction_error_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32, prediction_error: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
        self.prediction_error_sum += prediction_error;
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

    fn prediction_error(self) -> f32 {
        self.prediction_error_sum / self.cases.max(1) as f32
    }
}

#[derive(Clone, Copy)]
struct Snapshot {
    metrics: Metrics,
    edges: usize,
    spatial_edges: usize,
    temporal_edges: usize,
    regge: f32,
    deficit_regge: f32,
    free_energy: f32,
    criticality_distance: f32,
    temperature: f32,
    landauer_cost: f32,
    dvali: DvaliExtendedReport,
    causality_violations: usize,
}

fn main() {
    let lessons = lessons();
    let current = load_current_or_train(&lessons);
    let baseline_snapshot = snapshot(&current, &lessons, 0.0);

    let mut consolidated = current.clone();
    apply_n_portrait_temperature(&mut consolidated);
    apply_species_bound(&mut consolidated);
    apply_classicalization(&mut consolidated, &lessons);

    let landauer = landauer_from(&current, &consolidated);
    let consolidated_snapshot = snapshot(&consolidated, &lessons, landauer.cost);
    let keep = keep_consolidated(consolidated_snapshot, baseline_snapshot);
    if keep {
        let _ = consolidated.save_consolidated_state(KEPT_STATE);
    }

    println!("CDT-RQM consolidated Dvali substrate validation");
    println!(
        "lessons={} loaded_current={} kept_extensions=n_portrait_temperature,species_bound,classicalization kept_combined={} saved={}",
        lessons.len(),
        Path::new(KEPT_STATE).exists(),
        keep,
        keep
    );
    print_snapshot("current_dvali_substrate", baseline_snapshot);
    print_snapshot("consolidated_dvali_substrate", consolidated_snapshot);
    decision(
        "consolidated_dvali_extensions",
        keep,
        "keep value=n_portrait_temperature_species_bound_and_classicalization_improve_or_preserve_state",
        "discard value=consolidated_extensions_do_not_beat_current_state",
    );
}

fn apply_n_portrait_temperature(substrate: &mut CdtRqmUniverseSubstrate) {
    let n = substrate.hardware.active_edge_count().max(1) as f32;
    substrate.hardware.temperature = (1.0 / n.sqrt()).clamp(0.01, 1.0);
}

fn apply_species_bound(substrate: &mut CdtRqmUniverseSubstrate) {
    let species = dvali_extended_report(substrate, 0.0, 0.0)
        .species_count
        .max(1.0);
    let cutoff = 1.0 / species.sqrt();
    substrate.config.max_quantum_candidates =
        ((substrate.config.max_quantum_candidates as f32 * cutoff.max(0.25)).round() as usize)
            .clamp(16, substrate.config.max_quantum_candidates);
    substrate.software.config.activation_threshold =
        (substrate.software.config.activation_threshold * (1.0 + cutoff * 0.10)).min(0.08);
}

fn apply_classicalization(substrate: &mut CdtRqmUniverseSubstrate, lessons: &[Lesson]) {
    let metrics = evaluate(substrate, lessons);
    let radius = dvali_extended_report(substrate, metrics.leakage(), metrics.prediction_error())
        .classicalization_radius;
    if radius > 1.0 {
        substrate.hardware.temperature *= 0.85;
    }
}

fn snapshot(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    landauer_cost: f32,
) -> Snapshot {
    let metrics = evaluate(substrate, lessons);
    let dvali = dvali_extended_report(substrate, metrics.leakage(), metrics.prediction_error());
    let free_energy = unified_free_energy(
        substrate,
        metrics.prediction_error(),
        metrics.leakage(),
        UnifiedFreeEnergyConfig::default(),
    )
    .free_energy;
    Snapshot {
        metrics,
        edges: substrate.hardware.active_edge_count(),
        spatial_edges: substrate.hardware.active_spatial_edge_count(),
        temporal_edges: substrate.hardware.active_temporal_edge_count(),
        regge: substrate.hardware.regge_action(),
        deficit_regge: substrate.hardware.discrete_regge_deficit_action(),
        free_energy,
        criticality_distance: substrate.hardware.criticality_distance(),
        temperature: substrate.hardware.temperature,
        landauer_cost,
        dvali,
        causality_violations: substrate.hardware.causality_violations(),
    }
}

fn keep_consolidated(candidate: Snapshot, baseline: Snapshot) -> bool {
    if !preserves(candidate, baseline) {
        return false;
    }
    let no_geometry_regression = candidate.edges <= baseline.edges
        && candidate.deficit_regge <= baseline.deficit_regge + 0.001
        && candidate.free_energy <= baseline.free_energy + 0.001;
    let before_temp_gap = (baseline.temperature - baseline.dvali.n_portrait_temperature).abs();
    let after_temp_gap = (candidate.temperature - candidate.dvali.n_portrait_temperature).abs();
    let n_portrait_aligned = after_temp_gap <= 0.001 || after_temp_gap + 0.001 < before_temp_gap;
    let species_gain = candidate.metrics.leakage() + 0.001 < baseline.metrics.leakage()
        || candidate.free_energy + 0.001 < baseline.free_energy;
    let classicalization_gain =
        candidate.dvali.classicalization_radius + 0.001 < baseline.dvali.classicalization_radius;
    no_geometry_regression && n_portrait_aligned && species_gain && classicalization_gain
}

fn preserves(candidate: Snapshot, baseline: Snapshot) -> bool {
    candidate.metrics.accuracy() + 0.0001 >= baseline.metrics.accuracy()
        && candidate.metrics.leakage() <= baseline.metrics.leakage() + 0.0001
        && candidate.causality_violations == 0
}

fn evaluate(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&lesson.local, 1.0);
        let report = trial.step_from_boundary(OBSERVER, 0.0, &lesson.local);
        let expected = score(&report.collapse, &lesson.remote);
        let distractor = score(&report.collapse, &lesson.distractor);
        metrics.record(expected, distractor, report.cdt.prediction_error);
    }
    metrics
}

fn score(report: &CollapseReport, targets: &[usize]) -> f32 {
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn landauer_from(
    before: &CdtRqmUniverseSubstrate,
    after: &CdtRqmUniverseSubstrate,
) -> snga::cdt_rqm_experimental::LandauerReport {
    landauer_report(
        before.hardware.active_edge_count(),
        after.hardware.active_edge_count(),
        before.hardware.temperature,
    )
}

fn print_snapshot(label: &str, snapshot: Snapshot) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3} prediction_error={:.3} edges={} spatial={} temporal={} regge={:.3} deficit_regge={:.3} free_energy={:.3} criticality_distance={:.3} temp={:.4} landauer_cost={:.3} N={:.1} alpha={:.6} alphaN={:.3} T_N={:.4} depletion={:.4} lifetime={:.1} break_time={:.1} species={:.1} cutoff={:.4} memory_burden={:.3} classical_radius={:.3} causality_violations={}",
        label,
        snapshot.metrics.accuracy() * 100.0,
        snapshot.metrics.leakage() * 100.0,
        snapshot.metrics.margin(),
        snapshot.metrics.prediction_error(),
        snapshot.edges,
        snapshot.spatial_edges,
        snapshot.temporal_edges,
        snapshot.regge,
        snapshot.deficit_regge,
        snapshot.free_energy,
        snapshot.criticality_distance,
        snapshot.temperature,
        snapshot.landauer_cost,
        snapshot.dvali.occupation_number,
        snapshot.dvali.alpha_eff,
        snapshot.dvali.maximal_packing,
        snapshot.dvali.n_portrait_temperature,
        snapshot.dvali.depletion_rate,
        snapshot.dvali.evaporation_lifetime,
        snapshot.dvali.quantum_break_time,
        snapshot.dvali.species_count,
        snapshot.dvali.species_cutoff,
        snapshot.dvali.memory_burden,
        snapshot.dvali.classicalization_radius,
        snapshot.causality_violations,
    );
}

fn decision(label: &str, keep: bool, keep_message: &str, discard_message: &str) {
    println!(
        "decision_{}: {}",
        label,
        if keep { keep_message } else { discard_message }
    );
}

fn load_current_or_train(lessons: &[Lesson]) -> CdtRqmUniverseSubstrate {
    let mut substrate = substrate(epr_config(8));
    if Path::new(KEPT_STATE).exists() && substrate.load_consolidated_state(KEPT_STATE).is_ok() {
        return substrate;
    }
    train_epr(&mut substrate, lessons, 7);
    apply_n_portrait_temperature(&mut substrate);
    apply_species_bound(&mut substrate);
    apply_classicalization(&mut substrate, lessons);
    substrate
}

fn train_epr(substrate: &mut CdtRqmUniverseSubstrate, lessons: &[Lesson], epochs: usize) {
    for _ in 0..epochs {
        for lesson in lessons {
            substrate.hardware.clear_activity();
            substrate.train_observed_transition(OBSERVER, 0.0, &lesson.local, &lesson.remote, 1.0);
            for (&a, &b) in lesson.local.iter().zip(lesson.remote.iter()) {
                substrate.observe_entanglement_correlation(a, b, 0.45);
            }
        }
    }
}

fn lessons() -> Vec<Lesson> {
    [
        ("vanchurin", "madelung", "noise"),
        ("mera", "holography", "flat"),
        ("dvali", "criticality", "thermal"),
        ("wolfram", "causal_invariance", "branch"),
        ("graphity", "geometrogenesis", "complete"),
        ("landauer", "dissipation", "free"),
        ("page", "retention", "loss"),
        ("markov", "blanket", "external"),
    ]
    .into_iter()
    .map(|(local, remote, distractor)| Lesson {
        local: pattern(local, 0),
        remote: pattern(remote, 1),
        distractor: pattern(distractor, 1),
    })
    .collect()
}

fn pattern(label: &str, slice: usize) -> Vec<usize> {
    let mut out = (0..10)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            label.hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * NODES_PER_SLICE + (hasher.finish() as usize % NODES_PER_SLICE)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn substrate(entanglement: EntanglementConfig) -> CdtRqmUniverseSubstrate {
    let mut substrate = CdtRqmUniverseSubstrate::new(config());
    substrate.enable_entanglement(entanglement);
    substrate
}

fn config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice: NODES_PER_SLICE,
            initial_spatial_connectivity: 0.0002,
            initial_temporal_connectivity: 0.0001,
            target_spatial_degree: 4,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 10,
            seed: 87_301,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.10,
            phase_learning_rate: 0.24,
            coherence_learning_rate: 0.13,
            uncertainty_learning_rate: 0.11,
            amplitude_decay: 0.001,
            coherence_decay: 0.0005,
            uncertainty_recovery: 0.002,
            activation_threshold: 0.02,
        },
        max_quantum_candidates: 128,
        rqm_feedback_gain: 0.42,
    }
}

fn epr_config(max_links_per_node: usize) -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node,
        max_syncs_per_step: 512,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    }
}
