use crate::cdt_graphity::MeraScaleReport;
use crate::cdt_rqm::CdtRqmUniverseSubstrate;
use crate::relational_field::ObserverId;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub struct UnifiedFreeEnergyConfig {
    pub lambda_regge: f32,
    pub lambda_cosmological: f32,
    pub lambda_epr: f32,
    pub lambda_leakage: f32,
    pub lambda_causality: f32,
    pub lambda_criticality: f32,
}

impl Default for UnifiedFreeEnergyConfig {
    fn default() -> Self {
        Self {
            lambda_regge: 0.0015,
            lambda_cosmological: 0.005,
            lambda_epr: 0.04,
            lambda_leakage: 0.40,
            lambda_causality: 1.0,
            lambda_criticality: 0.03,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HolographicEprReport {
    pub region_entropy: f32,
    pub boundary_area: usize,
    pub area_law_ratio: f32,
    pub active_links_touching: usize,
}

#[derive(Clone, Debug, Default)]
pub struct MeraReport {
    pub scales: Vec<MeraScaleReport>,
    pub compression_gain: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CriticalityReport {
    pub index: f32,
    pub distance: f32,
    pub gap_estimate: f32,
    pub area_entropy: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UnifiedFreeEnergyReport {
    pub prediction_error: f32,
    pub regge_deficit: f32,
    pub cosmological_action: f32,
    pub epr_entropy: f32,
    pub leakage: f32,
    pub causality_violations: usize,
    pub criticality_distance: f32,
    pub free_energy: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LandauerReport {
    pub erased_bits: f32,
    pub cost: f32,
    pub temperature: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PageCurveReport {
    pub early_entropy: f32,
    pub page_entropy: f32,
    pub late_entropy: f32,
    pub retention: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MultiwayCausalReport {
    pub rqm_first_edges: usize,
    pub epr_first_edges: usize,
    pub rqm_first_regge: f32,
    pub epr_first_regge: f32,
    pub divergence: f32,
    pub causally_invariant: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OperationalErEprReport {
    pub topology_score: f32,
    pub epr_score: f32,
    pub indistinguishability: f32,
}

#[derive(Clone, Debug, Default)]
pub struct MarkovBlanketReport {
    pub internal: Vec<usize>,
    pub blanket: Vec<usize>,
    pub external: Vec<usize>,
    pub blanket_ratio: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DvaliExtendedReport {
    pub occupation_number: f32,
    pub alpha_eff: f32,
    pub maximal_packing: f32,
    pub n_portrait_temperature: f32,
    pub depletion_rate: f32,
    pub evaporation_lifetime: f32,
    pub quantum_break_time: f32,
    pub species_count: f32,
    pub species_cutoff: f32,
    pub memory_burden: f32,
    pub classicalization_radius: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ExperimentalPhysicsObservables {
    pub holographic_epr: HolographicEprReport,
    pub mera: MeraReport,
    pub criticality: CriticalityReport,
    pub free_energy: UnifiedFreeEnergyReport,
    pub landauer: LandauerReport,
    pub page_curve: PageCurveReport,
    pub multiway: MultiwayCausalReport,
    pub er_epr: OperationalErEprReport,
    pub markov_blanket: MarkovBlanketReport,
    pub dvali: DvaliExtendedReport,
    pub geometrogenesis_order: f32,
}

impl ExperimentalPhysicsObservables {
    pub fn from_substrate(
        substrate: &CdtRqmUniverseSubstrate,
        observer: ObserverId,
        observer_phase: f32,
        region: &[usize],
        leakage: f32,
        prediction_error: f32,
    ) -> Self {
        let free_energy_config = UnifiedFreeEnergyConfig::default();
        Self {
            holographic_epr: holographic_epr(substrate, region, 0.25),
            mera: mera_report(substrate, &[4, 8, 16, 32]),
            criticality: criticality_report(substrate),
            free_energy: unified_free_energy(
                substrate,
                prediction_error,
                leakage,
                free_energy_config,
            ),
            landauer: LandauerReport {
                erased_bits: 0.0,
                cost: 0.0,
                temperature: substrate.hardware.temperature,
            },
            page_curve: page_curve_report(substrate, region),
            multiway: multiway_causal_probe(substrate, observer, observer_phase, region),
            er_epr: operational_er_epr_probe(substrate, observer, observer_phase, region),
            markov_blanket: markov_blanket(substrate, observer, observer_phase, region),
            dvali: dvali_extended_report(substrate, leakage, prediction_error),
            geometrogenesis_order: substrate.hardware.geometrogenesis_order_parameter(),
        }
    }

    pub fn serialize_summary(&self) -> String {
        format!(
            "experimental_observables holographic_ratio={:.7} mera_gain={:.7} criticality_distance={:.7} free_energy={:.7} landauer_cost={:.7} page_retention={:.7} multiway_divergence={:.7} er_epr_indistinguishability={:.7} blanket_ratio={:.7} dvali_maximal_packing={:.7} dvali_memory_burden={:.7} geometrogenesis_order={:.7}\n",
            self.holographic_epr.area_law_ratio,
            self.mera.compression_gain,
            self.criticality.distance,
            self.free_energy.free_energy,
            self.landauer.cost,
            self.page_curve.retention,
            self.multiway.divergence,
            self.er_epr.indistinguishability,
            self.markov_blanket.blanket_ratio,
            self.dvali.maximal_packing,
            self.dvali.memory_burden,
            self.geometrogenesis_order
        )
    }
}

pub fn holographic_epr(
    substrate: &CdtRqmUniverseSubstrate,
    region: &[usize],
    area_coupling: f32,
) -> HolographicEprReport {
    let Some(field) = substrate.entanglement.as_ref() else {
        return HolographicEprReport::default();
    };
    HolographicEprReport {
        region_entropy: field.region_entropy(region),
        boundary_area: field.boundary_area(region),
        area_law_ratio: field.holographic_area_law_ratio(region, area_coupling),
        active_links_touching: field.active_links_touching(region),
    }
}

pub fn mera_report(substrate: &CdtRqmUniverseSubstrate, scales: &[usize]) -> MeraReport {
    let reports = scales
        .iter()
        .copied()
        .map(|scale| substrate.hardware.mera_scale_summary(scale))
        .collect::<Vec<_>>();
    let compression_gain = reports
        .iter()
        .map(|report| (1.0 - report.compression_ratio).max(0.0))
        .sum::<f32>()
        / reports.len().max(1) as f32;
    MeraReport {
        scales: reports,
        compression_gain,
    }
}

pub fn criticality_report(substrate: &CdtRqmUniverseSubstrate) -> CriticalityReport {
    let active_edges = substrate.hardware.active_edge_count().max(1) as f32;
    let boundary_area = substrate.hardware.active_spatial_edge_count().max(1) as f32;
    CriticalityReport {
        index: substrate.hardware.criticality_index(),
        distance: substrate.hardware.criticality_distance(),
        gap_estimate: 1.0 / active_edges,
        area_entropy: boundary_area * 0.25,
    }
}

pub fn unified_free_energy(
    substrate: &CdtRqmUniverseSubstrate,
    prediction_error: f32,
    leakage: f32,
    config: UnifiedFreeEnergyConfig,
) -> UnifiedFreeEnergyReport {
    let regge_deficit = substrate.hardware.discrete_regge_deficit_action();
    let lambda = auto_lambda(substrate);
    let cosmological_action = substrate.hardware.cosmological_regge_action(lambda);
    let epr_entropy = substrate
        .entanglement_summary()
        .map(|report| report.mean_entropy)
        .unwrap_or_default();
    let causality_violations = substrate.hardware.causality_violations();
    let criticality_distance = substrate.hardware.criticality_distance();
    let free_energy = prediction_error
        + config.lambda_regge * regge_deficit
        + config.lambda_cosmological * cosmological_action
        + config.lambda_epr * epr_entropy
        + config.lambda_leakage * leakage
        + config.lambda_causality * causality_violations as f32
        + config.lambda_criticality * criticality_distance;
    UnifiedFreeEnergyReport {
        prediction_error,
        regge_deficit,
        cosmological_action,
        epr_entropy,
        leakage,
        causality_violations,
        criticality_distance,
        free_energy,
    }
}

pub fn landauer_report(
    before_edges: usize,
    after_edges: usize,
    temperature: f32,
) -> LandauerReport {
    let erased_bits = before_edges.saturating_sub(after_edges) as f32;
    LandauerReport {
        erased_bits,
        cost: erased_bits * temperature.max(0.0) * std::f32::consts::LN_2,
        temperature,
    }
}

pub fn page_curve_report(substrate: &CdtRqmUniverseSubstrate, region: &[usize]) -> PageCurveReport {
    let Some(field) = substrate.entanglement.as_ref() else {
        return PageCurveReport::default();
    };
    let entropy = field.region_entropy(region);
    let area = field.boundary_area(region).max(1) as f32;
    let early_entropy = entropy * 0.5;
    let page_entropy = entropy.max(area.ln());
    let late_entropy = entropy.min(area.ln());
    PageCurveReport {
        early_entropy,
        page_entropy,
        late_entropy,
        retention: if page_entropy > f32::EPSILON {
            1.0 - (late_entropy / page_entropy - 0.5).abs().min(1.0)
        } else {
            1.0
        },
    }
}

pub fn multiway_causal_probe(
    substrate: &CdtRqmUniverseSubstrate,
    observer: ObserverId,
    observer_phase: f32,
    boundary: &[usize],
) -> MultiwayCausalReport {
    let mut rqm_first = substrate.clone();
    let rqm_report = rqm_first.step_from_boundary(observer, observer_phase, boundary);

    let mut epr_first = substrate.clone();
    let mut epr_boundary = boundary.to_vec();
    if let Some(field) = epr_first.entanglement.as_mut() {
        field.synchronize_candidates(boundary, &mut epr_boundary);
    }
    let epr_report = epr_first.step_from_boundary(observer, observer_phase, &epr_boundary);

    let rqm_first_edges = rqm_first.hardware.active_edge_count();
    let epr_first_edges = epr_first.hardware.active_edge_count();
    let rqm_first_regge = rqm_first.hardware.discrete_regge_deficit_action();
    let epr_first_regge = epr_first.hardware.discrete_regge_deficit_action();
    let edge_delta = rqm_first_edges.abs_diff(epr_first_edges) as f32
        / rqm_first_edges.max(epr_first_edges).max(1) as f32;
    let regge_delta =
        (rqm_first_regge - epr_first_regge).abs() / rqm_first_regge.max(epr_first_regge).max(1.0);
    let prediction_delta =
        (rqm_report.cdt.prediction_error - epr_report.cdt.prediction_error).abs();
    let divergence = (edge_delta + regge_delta + prediction_delta) / 3.0;
    MultiwayCausalReport {
        rqm_first_edges,
        epr_first_edges,
        rqm_first_regge,
        epr_first_regge,
        divergence,
        causally_invariant: divergence <= 0.05,
    }
}

pub fn operational_er_epr_probe(
    substrate: &CdtRqmUniverseSubstrate,
    observer: ObserverId,
    observer_phase: f32,
    boundary: &[usize],
) -> OperationalErEprReport {
    let mut epr_trial = substrate.clone();
    let epr_report = epr_trial.step_from_boundary(observer, observer_phase, boundary);
    let epr_score = epr_report.hardware_prediction_score;

    let mut topology_trial = substrate.clone();
    let mut topological_boundary = boundary.to_vec();
    let predictions = topology_trial
        .hardware
        .predict_next(boundary, boundary.len().max(1) * 2);
    for (candidate, _) in predictions {
        if !topological_boundary.contains(&candidate) {
            topological_boundary.push(candidate);
        }
    }
    let topology_report =
        topology_trial.step_from_boundary(observer, observer_phase, &topological_boundary);
    let topology_score = topology_report.hardware_prediction_score;
    let indistinguishability = 1.0 - (topology_score - epr_score).abs().min(1.0);

    OperationalErEprReport {
        topology_score,
        epr_score,
        indistinguishability,
    }
}

pub fn markov_blanket(
    substrate: &CdtRqmUniverseSubstrate,
    observer: ObserverId,
    observer_phase: f32,
    boundary: &[usize],
) -> MarkovBlanketReport {
    let mut trial = substrate.clone();
    let collapse = trial.software.observe_pattern(
        observer,
        boundary,
        observer_phase,
        boundary.len().max(1) * 2,
    );
    let internal = compact(boundary);
    let blanket = compact(
        &collapse
            .candidates
            .iter()
            .map(|candidate| candidate.agent)
            .collect::<Vec<_>>(),
    );
    let internal_set = internal.iter().copied().collect::<HashSet<_>>();
    let blanket_set = blanket.iter().copied().collect::<HashSet<_>>();
    let mut external = Vec::new();
    for (candidate, _) in substrate
        .hardware
        .predict_next(boundary, boundary.len().max(1) * 4)
    {
        if !internal_set.contains(&candidate) && !blanket_set.contains(&candidate) {
            external.push(candidate);
        }
    }
    let external = compact(&external);
    let blanket_ratio =
        blanket.len() as f32 / (internal.len() + blanket.len() + external.len()).max(1) as f32;

    MarkovBlanketReport {
        internal,
        blanket,
        external,
        blanket_ratio,
    }
}

pub fn auto_lambda(substrate: &CdtRqmUniverseSubstrate) -> f32 {
    let volume = substrate.hardware.tetrahedra.len().max(1) as f32;
    let curvature_density = substrate.hardware.discrete_regge_deficit_action() / volume;
    (0.05 / (1.0 + curvature_density / 16.0)).clamp(0.005, 0.05)
}

pub fn dvali_extended_report(
    substrate: &CdtRqmUniverseSubstrate,
    leakage: f32,
    prediction_error: f32,
) -> DvaliExtendedReport {
    let occupation_number = substrate.hardware.active_edge_count().max(1) as f32;
    let alpha_eff = 1.0 / occupation_number;
    let maximal_packing = alpha_eff * occupation_number;
    let n_portrait_temperature = 1.0 / occupation_number.sqrt();
    let depletion_rate = 1.0 / occupation_number.sqrt();
    let evaporation_lifetime = occupation_number.powf(1.5);
    let species_count = (substrate.relation_count() as f32 / occupation_number.sqrt())
        .max(1.0)
        .ceil();
    let radius = substrate.hardware.config.slices.max(1) as f32;
    let quantum_break_time = radius * occupation_number / species_count;
    let species_cutoff = 1.0 / species_count.sqrt();
    let useful_memory = substrate.relation_count() as f32 * (1.0 - leakage).max(0.0);
    let capacity = occupation_number + useful_memory.max(1.0);
    let memory_burden = (useful_memory / capacity).clamp(0.0, 1.0);
    let classicalization_radius = (prediction_error
        + substrate.hardware.temperature
        + substrate.hardware.criticality_distance())
    .max(0.0)
    .sqrt();

    DvaliExtendedReport {
        occupation_number,
        alpha_eff,
        maximal_packing,
        n_portrait_temperature,
        depletion_rate,
        evaporation_lifetime,
        quantum_break_time,
        species_count,
        species_cutoff,
        memory_burden,
        classicalization_radius,
    }
}

fn compact(values: &[usize]) -> Vec<usize> {
    let mut out = values.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}
