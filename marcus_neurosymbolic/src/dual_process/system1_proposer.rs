//! Módulo de Sistema 1: Generador Heurístico y Asociativo (Proponente Rápido).
//!
//! Basado en Gary Marcus (Rebooting AI, Cap. 2 y The Next Decade in AI, 2020):
//! "System 1 is fast, intuitive, and statistical. It proposes hypotheses based on pattern
//! matching and associations, but lacks world models and is prone to confident hallucinations."

use serde::{Deserialize, Serialize};

/// Hipótesis candidata generada por el Sistema 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateHypothesis {
    pub id: String,
    pub claim: String,
    /// Confianza estadística o asociativa asignada por el Sistema 1 (0.0 a 1.0).
    pub associative_confidence: f64,
    /// Proposición estructurada asociada (si el proponente pudo parsearla).
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl CandidateHypothesis {
    pub fn new(
        id: impl Into<String>,
        claim: impl Into<String>,
        confidence: f64,
        subj: impl Into<String>,
        pred: impl Into<String>,
        obj: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            claim: claim.into(),
            associative_confidence: confidence,
            subject: subj.into(),
            predicate: pred.into(),
            object: obj.into(),
        }
    }
}

/// Proponente asociativo del Sistema 1.
#[derive(Debug, Clone, Default)]
pub struct System1Proposer {
    /// Base de correlaciones y n-gramas estadísticos aprendidos del corpus.
    corpus_associations: Vec<CandidateHypothesis>,
}

impl System1Proposer {
    pub fn new() -> Self {
        Self {
            corpus_associations: Vec::new(),
        }
    }

    /// Registra una correlación observada en el entrenamiento estadístico.
    pub fn add_association(&mut self, hypothesis: CandidateHypothesis) {
        self.corpus_associations.push(hypothesis);
    }

    /// Propone los mejores candidatos basados en coincidencia de patrones con el query.
    pub fn propose_candidates(&self, query: &str, top_k: usize) -> Vec<CandidateHypothesis> {
        let q_lower = query.to_lowercase();
        let mut scored: Vec<(f64, &CandidateHypothesis)> = self
            .corpus_associations
            .iter()
            .map(|hyp| {
                let text = format!("{} {} {}", hyp.subject, hyp.predicate, hyp.object).to_lowercase();
                // Similitud léxica superficial Jaccard/overlap de palabras
                let q_tokens: std::collections::HashSet<&str> = q_lower.split_whitespace().collect();
                let h_tokens: std::collections::HashSet<&str> = text.split_whitespace().collect();
                let intersection = q_tokens.intersection(&h_tokens).count() as f64;
                let union = q_tokens.union(&h_tokens).count().max(1) as f64;
                let overlap = intersection / union;
                // Puntuación combinada de similitud de superficie + confianza asociativa previa
                let score = 0.5 * overlap + 0.5 * hyp.associative_confidence;
                (score, hyp)
            })
            .collect();

        // Ordenar descendentemente por puntuación
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(_, h)| h.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system1_proposal() {
        let mut sys1 = System1Proposer::new();
        sys1.add_association(CandidateHypothesis::new(
            "h1",
            "Birds can fly",
            0.95,
            "Bird",
            "CanFly",
            "True",
        ));
        sys1.add_association(CandidateHypothesis::new(
            "h2",
            "Penguins can fly (spurious high co-occurrence in cartoons)",
            0.82,
            "Penguin",
            "CanFly",
            "True",
        ));

        let candidates = sys1.propose_candidates("Penguin fly", 2);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].subject, "Penguin");
    }
}
