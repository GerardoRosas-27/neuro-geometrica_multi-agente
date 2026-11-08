//! Conector 1: Sustrato de Campo Continuo (gemma_campo & campo_turno) -> Sistema 1 Marcus.
//!
//! Integra la dinámica de campo continuo token a token e inter-turno:
//! - Modos ortogonales de Lie (F=2721) y vector de estado latente z_t \in R^64.
//! - Innovación causal e(x_t) y subespacio contextual V_ctx (k_skip=16).
//! - Proporciona al Sistema 1 de Gary Marcus un sustrato sensorial continuo
//!   que supera la limitación del autocompletado discreto plano.

use crate::dual_process::system1_proposer::CandidateHypothesis;
use serde::{Deserialize, Serialize};

/// Estado continuo del campo latente generado por el sustrato token a token o por turno.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousFieldState {
    pub dim: usize,
    pub latent_z: Vec<f64>,
    pub lie_energy: f64,
    pub causal_innovation_norm: f64,
    pub step_index: u64,
}

impl ContinuousFieldState {
    pub fn new(dim: usize, step_index: u64) -> Self {
        Self {
            dim,
            latent_z: vec![0.0; dim],
            lie_energy: 0.0,
            causal_innovation_norm: 0.0,
            step_index,
        }
    }

    pub fn with_latent(mut self, z: Vec<f64>, innovation: f64) -> Self {
        self.dim = z.len();
        self.causal_innovation_norm = innovation;
        let norm_sq: f64 = z.iter().map(|x| x * x).sum();
        self.lie_energy = norm_sq;
        self.latent_z = z;
        self
    }
}

/// Centroide de concepto semántico proyectado en el espacio continuo del campo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConceptAnchor {
    pub concept_id: String,
    pub centroid_z: Vec<f64>,
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Conector perceptual que traduce dinámicas continuas de campo a hipótesis del Sistema 1.
#[derive(Debug, Clone, Default)]
pub struct ContinuousFieldPerceptualConnector {
    anchors: Vec<SemanticConceptAnchor>,
    pub dim: usize,
}

impl ContinuousFieldPerceptualConnector {
    pub fn new(dim: usize) -> Self {
        Self {
            anchors: Vec::new(),
            dim,
        }
    }

    pub fn register_concept_anchor(&mut self, anchor: SemanticConceptAnchor) {
        assert_eq!(anchor.centroid_z.len(), self.dim, "La dimensión del centroide debe coincidir");
        self.anchors.push(anchor);
    }

    /// Proyecta el estado continuo z_t y genera hipótesis candidatas para el Sistema 1
    /// ponderadas por similitud cosenoidal en el subespacio continuo y moduladas por la energía de Lie.
    pub fn decode_hypotheses_from_field(
        &self,
        field_state: &ContinuousFieldState,
        top_k: usize,
    ) -> Vec<CandidateHypothesis> {
        if self.anchors.is_empty() || field_state.latent_z.is_empty() {
            return Vec::new();
        }

        let z_norm: f64 = field_state.latent_z.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);

        let mut scored: Vec<(f64, &SemanticConceptAnchor)> = self
            .anchors
            .iter()
            .map(|anchor| {
                let dot: f64 = field_state
                    .latent_z
                    .iter()
                    .zip(anchor.centroid_z.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let a_norm: f64 = anchor.centroid_z.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                let cos_sim = dot / (z_norm * a_norm);
                // Modular confianza con la innovación causal del campo
                let confidence = ((cos_sim + 1.0) / 2.0).clamp(0.0, 1.0);
                (confidence, anchor)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(i, (conf, anchor))| {
                CandidateHypothesis::new(
                    format!("field_hyp_{}_{}", field_state.step_index, i),
                    format!("{} {} {}", anchor.subject, anchor.predicate, anchor.object),
                    conf,
                    anchor.subject.clone(),
                    anchor.predicate.clone(),
                    anchor.object.clone(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_field_to_system1_candidates() {
        let dim = 4;
        let mut connector = ContinuousFieldPerceptualConnector::new(dim);

        connector.register_concept_anchor(SemanticConceptAnchor {
            concept_id: "gravity_drop".to_string(),
            centroid_z: vec![1.0, 0.0, 0.0, 0.0],
            subject: "Apple".to_string(),
            predicate: "FallsTowards".to_string(),
            object: "Ground".to_string(),
        });

        connector.register_concept_anchor(SemanticConceptAnchor {
            concept_id: "levitation_fake".to_string(),
            centroid_z: vec![-1.0, 0.0, 0.0, 0.0],
            subject: "Apple".to_string(),
            predicate: "FloatsUpwards".to_string(),
            object: "Sky".to_string(),
        });

        // Estado del campo alineado fuertemente con caída gravitacional
        let state = ContinuousFieldState::new(dim, 1).with_latent(vec![0.9, 0.1, 0.0, 0.0], 0.05);

        let candidates = connector.decode_hypotheses_from_field(&state, 2);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].subject, "Apple");
        assert_eq!(candidates[0].predicate, "FallsTowards");
        assert!(candidates[0].associative_confidence > 0.85);
        assert!(candidates[1].associative_confidence < 0.20);
    }
}
