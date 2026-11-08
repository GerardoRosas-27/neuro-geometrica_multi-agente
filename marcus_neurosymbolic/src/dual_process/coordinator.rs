//! Módulo de Coordinación Bidireccional (Dual-Process Coordinator).
//!
//! Orquesta el ciclo completo de Sistema 1 (proposición rápida asociativa)
//! y Sistema 2 (deliberación y verificación simbólica estricta).

use crate::dual_process::system1_proposer::{CandidateHypothesis, System1Proposer};
use crate::dual_process::system2_deliberator::{DeliberationVerdict, System2Deliberator};
use serde::{Deserialize, Serialize};

/// Resultado coordinado de la cognición dual.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualProcessResult {
    pub query: String,
    pub selected_hypothesis: Option<CandidateHypothesis>,
    pub verdict: Option<DeliberationVerdict>,
    pub rejected_hypotheses: Vec<(CandidateHypothesis, String)>,
    pub cycles_elapsed: usize,
}

pub struct DualProcessCoordinator {
    pub max_candidate_evaluations: usize,
}

impl DualProcessCoordinator {
    pub fn new() -> Self {
        Self {
            max_candidate_evaluations: 5,
        }
    }

    /// Ejecuta el ciclo coordinado neuro-simbólico.
    pub fn execute_cycle(
        &self,
        query: &str,
        system1: &System1Proposer,
        system2: &mut System2Deliberator,
    ) -> DualProcessResult {
        let candidates = system1.propose_candidates(query, self.max_candidate_evaluations);
        let mut rejected = Vec::new();

        for candidate in candidates {
            let verdict = system2.deliberate(&candidate);
            match &verdict {
                DeliberationVerdict::Approved { .. } => {
                    let cycles = rejected.len() + 1;
                    return DualProcessResult {
                        query: query.to_string(),
                        selected_hypothesis: Some(candidate),
                        verdict: Some(verdict),
                        rejected_hypotheses: rejected,
                        cycles_elapsed: cycles,
                    };
                }
                DeliberationVerdict::Vetoed { reason, .. } => {
                    rejected.push((candidate, reason.clone()));
                }
            }
        }

        // Si todos los candidatos del Sistema 1 fueron vetados por imposibilidad lógica/física
        DualProcessResult {
            query: query.to_string(),
            selected_hypothesis: None,
            verdict: None,
            rejected_hypotheses: rejected,
            cycles_elapsed: self.max_candidate_evaluations,
        }
    }
}
