use crate::cdt_graphity::{CdtGraphityConfig, CdtGraphityStepReport, CdtGraphitySubstrate};
use crate::entanglement::{EntanglementConfig, EntanglementField, EntanglementReport};
use crate::relational_field::{
    CollapseReport, ObserverId, RelationalFieldConfig, RelationalFieldSubstrate,
};
use crate::relational_guidance::RelationalGuidanceEngine;
use std::fs;
use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct CdtRqmConfig {
    pub cdt: CdtGraphityConfig,
    pub rqm: RelationalFieldConfig,
    pub max_quantum_candidates: usize,
    pub rqm_feedback_gain: f32,
}

impl Default for CdtRqmConfig {
    fn default() -> Self {
        Self {
            cdt: CdtGraphityConfig::default(),
            rqm: RelationalFieldConfig::default(),
            max_quantum_candidates: 8,
            rqm_feedback_gain: 0.35,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CdtRqmStepReport {
    pub cdt: CdtGraphityStepReport,
    pub collapse: CollapseReport,
    pub observer: ObserverId,
    pub observer_phase: f32,
    pub expected_from_rqm: Vec<usize>,
    pub hardware_prediction_score: f32,
    pub software_candidates: usize,
    pub entanglement: Option<EntanglementReport>,
}

#[derive(Clone, Debug)]
pub struct CdtRqmAnnealReport {
    pub attempts: usize,
    pub accepted: usize,
    pub initial_accuracy: f32,
    pub final_accuracy: f32,
    pub initial_leakage: f32,
    pub final_leakage: f32,
    pub initial_regge: f32,
    pub final_regge: f32,
    pub initial_edges: usize,
    pub final_edges: usize,
    pub causality_violations: usize,
}

#[derive(Clone, Debug)]
pub struct CdtRqmUniverseSubstrate {
    pub hardware: CdtGraphitySubstrate,
    pub software: RelationalFieldSubstrate,
    pub entanglement: Option<EntanglementField>,
    guidance: RelationalGuidanceEngine,
    pub config: CdtRqmConfig,
}

impl CdtRqmUniverseSubstrate {
    pub fn new(config: CdtRqmConfig) -> Self {
        Self {
            hardware: CdtGraphitySubstrate::graphity_hot_start(config.cdt),
            software: RelationalFieldSubstrate::new(config.rqm),
            entanglement: None,
            guidance: RelationalGuidanceEngine::new(),
            config,
        }
    }

    pub fn enable_entanglement(&mut self, config: EntanglementConfig) {
        self.entanglement = Some(EntanglementField::new(config));
    }

    pub fn entanglement_summary(&self) -> Option<EntanglementReport> {
        self.entanglement.as_ref().map(EntanglementField::summary)
    }

    pub fn observe_entanglement_correlation(&mut self, a: usize, b: usize, benefit: f32) -> bool {
        self.entanglement
            .as_mut()
            .map(|field| field.observe_correlation(a, b, benefit))
            .unwrap_or(false)
    }

    pub fn inject_entanglement_conflict(
        &mut self,
        a: usize,
        b: usize,
    ) -> Option<EntanglementReport> {
        self.entanglement
            .as_mut()
            .map(|field| field.inject_conflict(a, b))
    }

    pub fn train_observed_transition(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        cause: &[usize],
        effect: &[usize],
        success: f32,
    ) -> CdtRqmStepReport {
        for &source in cause {
            for &target in effect {
                self.software
                    .reinforce_relation(observer, source, target, observer_phase, success);
            }
        }
        self.hardware.inject_pattern(cause, 1.0);
        let mut report = self.step_from_boundary(observer, observer_phase, cause);

        // The observed effect acts as corrective feedback when the software proposes
        // an incomplete future, analogous to a local RQM information update.
        let proposed = report
            .expected_from_rqm
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let missed = effect
            .iter()
            .copied()
            .filter(|idx| !proposed.contains(idx))
            .collect::<Vec<_>>();
        if !missed.is_empty() {
            for &source in cause {
                for &target in &missed {
                    self.software.reinforce_relation(
                        observer,
                        source,
                        target,
                        observer_phase,
                        self.config.rqm_feedback_gain,
                    );
                }
            }
            self.hardware.inject_pattern(cause, 1.0);
            report = self.step_from_boundary(observer, observer_phase, cause);
        }
        report
    }

    pub fn migrate_causal_edges<I>(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        edges: I,
        min_weight: f32,
    ) -> usize
    where
        I: IntoIterator<Item = (usize, usize, f32)>,
    {
        let mut migrated = 0;
        for (source, target, weight) in edges {
            if weight < min_weight {
                continue;
            }
            self.software.reinforce_relation(
                observer,
                source,
                target,
                observer_phase,
                weight.clamp(0.0, 1.0),
            );
            if self
                .hardware
                .reinforce_temporal_link(source, target, weight.clamp(0.0, 1.0))
            {
                migrated += 1;
            }
        }
        migrated
    }

    pub fn train_binary_sequence(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        sequence: &[Vec<usize>],
        success: f32,
    ) -> usize {
        let mut transitions = 0;
        for window in sequence.windows(2) {
            self.train_observed_transition(
                observer,
                observer_phase,
                &window[0],
                &window[1],
                success,
            );
            transitions += 1;
        }
        transitions
    }

    pub fn anneal_after_migration(
        &mut self,
        validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
        attempts: usize,
    ) -> CdtRqmAnnealReport {
        let (initial_accuracy, initial_leakage) = self.validation_scores(validation);
        let initial_regge = self.hardware.regge_action();
        let initial_edges = self
            .hardware
            .edges
            .iter()
            .filter(|edge| edge.active)
            .count();
        let protected_edges = validation
            .iter()
            .flat_map(|(_, _, cue, expected, _)| {
                cue.iter()
                    .flat_map(move |source| expected.iter().map(move |target| (*source, *target)))
            })
            .collect::<Vec<_>>();

        let mut accepted = 0;
        let mut best = self.clone();
        let mut best_accuracy = initial_accuracy;
        let mut best_leakage = initial_leakage;
        let mut best_regge = initial_regge;
        let mut best_edges = initial_edges;
        let lambda = auto_cosmological_lambda(&self.hardware);
        let mut best_lambda_action = self.hardware.cosmological_regge_action(lambda);

        for _ in 0..attempts {
            for move_kind in [0_u8, 1, 2] {
                let mut candidate = best.clone();
                match move_kind {
                    0 => {
                        candidate.hardware.anneal_geometry_step(&protected_edges);
                    }
                    1 => {
                        candidate.hardware.hawking_radiation_step(&protected_edges);
                    }
                    _ => {
                        candidate
                            .hardware
                            .cosmological_constant_step(&protected_edges, lambda);
                    }
                }
                let (accuracy, leakage) = candidate.validation_scores(validation);
                let regge = candidate.hardware.regge_action();
                let lambda_action = candidate.hardware.cosmological_regge_action(lambda);
                let edges = candidate
                    .hardware
                    .edges
                    .iter()
                    .filter(|edge| edge.active)
                    .count();
                let preserves_memory =
                    accuracy + 0.0001 >= best_accuracy && leakage <= best_leakage + 0.0001;
                let improves_geometry =
                    regge < best_regge || edges < best_edges || lambda_action < best_lambda_action;
                if preserves_memory && improves_geometry {
                    best = candidate;
                    best_accuracy = accuracy;
                    best_leakage = leakage;
                    best_regge = regge;
                    best_lambda_action = lambda_action;
                    best_edges = edges;
                    accepted += 1;
                }
            }
        }

        *self = best;
        CdtRqmAnnealReport {
            attempts,
            accepted,
            initial_accuracy,
            final_accuracy: best_accuracy,
            initial_leakage,
            final_leakage: best_leakage,
            initial_regge,
            final_regge: best_regge,
            initial_edges,
            final_edges: best_edges,
            causality_violations: self.hardware.causality_violations(),
        }
    }

    pub fn hawking_radiation_after_migration(
        &mut self,
        validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
        attempts: usize,
    ) -> CdtRqmAnnealReport {
        let (initial_accuracy, initial_leakage) = self.validation_scores(validation);
        let initial_regge = self.hardware.regge_action();
        let initial_edges = self
            .hardware
            .edges
            .iter()
            .filter(|edge| edge.active)
            .count();
        let protected_edges = validation
            .iter()
            .flat_map(|(_, _, cue, expected, _)| {
                cue.iter()
                    .flat_map(move |source| expected.iter().map(move |target| (*source, *target)))
            })
            .collect::<Vec<_>>();

        let mut accepted = 0;
        let mut best = self.clone();
        let mut best_accuracy = initial_accuracy;
        let mut best_leakage = initial_leakage;
        let mut best_regge = initial_regge;
        let mut best_edges = initial_edges;

        for _ in 0..attempts {
            let mut candidate = best.clone();
            candidate.hardware.hawking_radiation_step(&protected_edges);
            let (accuracy, leakage) = candidate.validation_scores(validation);
            let regge = candidate.hardware.regge_action();
            let edges = candidate
                .hardware
                .edges
                .iter()
                .filter(|edge| edge.active)
                .count();
            let preserves_memory =
                accuracy + 0.0001 >= best_accuracy && leakage <= best_leakage + 0.0001;
            let improves_geometry = regge < best_regge || edges < best_edges;
            if preserves_memory && improves_geometry {
                best = candidate;
                best_accuracy = accuracy;
                best_leakage = leakage;
                best_regge = regge;
                best_edges = edges;
                accepted += 1;
            }
        }

        *self = best;
        CdtRqmAnnealReport {
            attempts,
            accepted,
            initial_accuracy,
            final_accuracy: best_accuracy,
            initial_leakage,
            final_leakage: best_leakage,
            initial_regge,
            final_regge: best_regge,
            initial_edges,
            final_edges: best_edges,
            causality_violations: self.hardware.causality_violations(),
        }
    }

    pub fn step(&mut self, observer: ObserverId, observer_phase: f32) -> CdtRqmStepReport {
        let active = self.hardware.active_pattern();
        self.step_from_boundary(observer, observer_phase, &active)
    }

    pub fn step_from_boundary(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        boundary: &[usize],
    ) -> CdtRqmStepReport {
        let mut collapse = self.software.observe_pattern(
            observer,
            boundary,
            observer_phase,
            self.config.max_quantum_candidates,
        );
        if !is_typed_memory_observer(observer) {
            self.guidance.apply(&self.hardware, &mut collapse);
        }
        let mut expected_from_rqm = collapse
            .candidates
            .iter()
            .map(|candidate| candidate.agent)
            .collect::<Vec<_>>();
        let software_candidates = collapse.candidates.len();
        let entanglement = self
            .entanglement
            .as_mut()
            .map(|field| field.synchronize_candidates(boundary, &mut expected_from_rqm));
        let cdt = self.hardware.step(&expected_from_rqm);
        let hardware_prediction_score = if expected_from_rqm.is_empty() {
            0.0
        } else {
            1.0 - cdt.prediction_error
        };
        CdtRqmStepReport {
            cdt,
            collapse,
            observer,
            observer_phase,
            expected_from_rqm,
            hardware_prediction_score,
            software_candidates,
            entanglement,
        }
    }

    pub fn relation_count(&self) -> usize {
        self.software.relation_count()
    }

    pub fn grow_foliated_block(&mut self, block_slices: usize) -> usize {
        self.hardware.add_foliated_block(block_slices)
    }

    pub fn save_consolidated_state<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.serialize_consolidated_state())
    }

    pub fn load_consolidated_state<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let contents = fs::read_to_string(path)?;
        self.apply_consolidated_state(&contents)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))
    }

    pub fn serialize_consolidated_state(&self) -> String {
        let active_edges = self
            .hardware
            .edges
            .iter()
            .filter(|edge| edge.active)
            .count();
        let mut out = String::new();
        out.push_str("CDT_RQM_EPR_CONSOLIDATED_STATE_V1\n");
        out.push_str(&format!(
            "summary relations={} active_edges={} regge={:.7} temperature={:.7} causality_violations={}\n",
            self.relation_count(),
            active_edges,
            self.hardware.regge_action(),
            self.hardware.temperature,
            self.hardware.causality_violations()
        ));
        if let Some(report) = self.entanglement_summary() {
            out.push_str(&format!(
                "entanglement active_links={} mean_coherence={:.7} mean_entropy={:.7}\n",
                report.active_links, report.mean_coherence, report.mean_entropy
            ));
        }
        out.push_str("hardware_begin\n");
        out.push_str(&self.hardware.serialize_persistent_state());
        out.push_str("hardware_end\n");
        out.push_str("software_begin\n");
        out.push_str(&self.software.serialize_persistent_state());
        out.push_str("software_end\n");
        if let Some(entanglement) = &self.entanglement {
            out.push_str("entanglement_begin\n");
            out.push_str(&entanglement.serialize_persistent_state());
            out.push_str("entanglement_end\n");
        }
        out.push_str("observables_begin\n");
        let active = self.hardware.active_pattern();
        out.push_str(
            &crate::cdt_rqm_experimental::ExperimentalPhysicsObservables::from_substrate(
                self,
                ObserverId(0),
                0.0,
                &active,
                0.0,
                0.0,
            )
            .serialize_summary(),
        );
        out.push_str("observables_end\n");
        out.push_str("end\n");
        out
    }

    pub fn apply_consolidated_state(&mut self, contents: &str) -> Result<(), String> {
        let mut lines = contents.lines();
        let version = lines.next();
        if version != Some("CDT_RQM_EPR_CONSOLIDATED_STATE_V1") {
            return Err("version consolidada CDT-RQM invalida".to_string());
        }
        let _summary = lines.next().ok_or("falta resumen consolidado")?;
        let rest = lines.collect::<Vec<_>>().join("\n");
        let hardware =
            section(&rest, "hardware_begin", "hardware_end").ok_or("falta seccion hardware")?;
        let software =
            section(&rest, "software_begin", "software_end").ok_or("falta seccion software")?;
        self.hardware.apply_persistent_state(&hardware)?;
        self.software.apply_persistent_state(&software)?;
        if let Some(entanglement) = section(&rest, "entanglement_begin", "entanglement_end") {
            let mut field = EntanglementField::new(EntanglementConfig::default());
            field.apply_persistent_state(&entanglement)?;
            self.entanglement = Some(field);
        }
        Ok(())
    }

    fn validation_scores(
        &self,
        validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
    ) -> (f32, f32) {
        let mut trial = self.clone();
        let mut correct = 0_usize;
        let mut leakage_sum = 0.0_f32;
        for (observer, phase, cue, expected, distractor) in validation {
            trial.hardware.clear_activity();
            trial.hardware.inject_pattern(cue, 1.0);
            let report = trial.step_from_boundary(*observer, *phase, cue);
            let expected_score = report
                .collapse
                .candidates
                .iter()
                .filter(|candidate| expected.contains(&candidate.agent))
                .map(|candidate| candidate.score)
                .sum::<f32>();
            let distractor_score = report
                .collapse
                .candidates
                .iter()
                .filter(|candidate| distractor.contains(&candidate.agent))
                .map(|candidate| candidate.score)
                .sum::<f32>();
            correct += usize::from(expected_score > distractor_score);
            let total = expected_score + distractor_score;
            leakage_sum += if total > f32::EPSILON {
                distractor_score / total
            } else {
                1.0
            };
        }
        (
            correct as f32 / validation.len().max(1) as f32,
            leakage_sum / validation.len().max(1) as f32,
        )
    }
}

fn section(contents: &str, begin: &str, end: &str) -> Option<String> {
    let start = contents.find(begin)? + begin.len();
    let tail = &contents[start..];
    let stop = tail.find(end)?;
    Some(tail[..stop].trim_matches('\n').to_string())
}

fn auto_cosmological_lambda(hardware: &CdtGraphitySubstrate) -> f32 {
    let volume = hardware.tetrahedra.len().max(1) as f32;
    let curvature_density = hardware.regge_action() / volume;
    (0.05 / (1.0 + curvature_density / 16.0)).clamp(0.005, 0.05)
}

fn is_typed_memory_observer(observer: ObserverId) -> bool {
    matches!(observer.0, 261_001..=261_004)
}
