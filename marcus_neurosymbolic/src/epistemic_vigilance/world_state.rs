//! Módulo de Estado del Mundo Epistémico (World State Tracker).
//!
//! Basado en Gary Marcus (Taming Silicon Valley, 2024 y Rebooting AI, Cap. 7):
//! "Without a persistent, verifiable model of the world, language models wander
//! through plausible-sounding confabulations. An epistemic engine must maintain
//! ground-truth states and track certainty levels."

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Nivel de certeza epistémica de una proposición.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EpistemicCertainty {
    /// Especulación no comprobada o conjetura estadística.
    Speculative,
    /// Observado directamente por sensores fidedignos.
    ObservedFact,
    /// Deducido analíticamente mediante reglas algebraicas o leyes físicas invariantes.
    AnalyticProof,
}

/// Hecho epistémico registrado en el estado del mundo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthFact {
    pub key: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub certainty: EpistemicCertainty,
    pub timestamp_step: u64,
}

/// Estado del mundo persistente para anclar la verdad objetiva.
#[derive(Debug, Clone, Default)]
pub struct WorldStateTracker {
    current_step: u64,
    facts: HashMap<String, GroundTruthFact>,
}

impl WorldStateTracker {
    pub fn new() -> Self {
        Self {
            current_step: 0,
            facts: HashMap::new(),
        }
    }

    pub fn advance_step(&mut self) {
        self.current_step += 1;
    }

    pub fn insert_fact(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        certainty: EpistemicCertainty,
    ) {
        let s = subject.into();
        let p = predicate.into();
        let o = object.into();
        let key = format!("{}:{}:{}", s, p, o);
        self.facts.insert(
            key.clone(),
            GroundTruthFact {
                key,
                subject: s,
                predicate: p,
                object: o,
                certainty,
                timestamp_step: self.current_step,
            },
        );
    }

    pub fn fact_exists(&self, subject: &str, predicate: &str, object: &str) -> bool {
        let key = format!("{}:{}:{}", subject, predicate, object);
        self.facts.contains_key(&key)
    }

    pub fn get_fact(&self, subject: &str, predicate: &str, object: &str) -> Option<&GroundTruthFact> {
        let key = format!("{}:{}:{}", subject, predicate, object);
        self.facts.get(&key)
    }

    pub fn all_facts(&self) -> impl Iterator<Item = &GroundTruthFact> {
        self.facts.values()
    }
}
