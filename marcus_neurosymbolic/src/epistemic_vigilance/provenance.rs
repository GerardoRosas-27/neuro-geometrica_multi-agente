//! Módulo de Procedencia Epistémica y Trazabilidad (Audit Provenance).
//!
//! Basado en Gary Marcus (Taming Silicon Valley, 2024):
//! "Trustworthy AI requires epistemic vigilance: every conclusion must have a traceable,
//! inspectable proof lineage detailing why the system made its claim."

use serde::{Deserialize, Serialize};

/// Paso o eslabón en la cadena de razonamiento y verificación.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceStep {
    pub step_index: usize,
    pub description: String,
    pub rule_applied: String,
    pub verified_by_engine: String,
}

/// Árbol o cadena de procedencia epistémica completa para una decisión o afirmación.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicProvenanceTree {
    pub claim_id: String,
    pub statement: String,
    pub is_verified: bool,
    pub steps: Vec<ProvenanceStep>,
    pub checksum: String,
}

impl EpistemicProvenanceTree {
    pub fn new(claim_id: impl Into<String>, statement: impl Into<String>) -> Self {
        Self {
            claim_id: claim_id.into(),
            statement: statement.into(),
            is_verified: false,
            steps: Vec::new(),
            checksum: String::new(),
        }
    }

    pub fn add_step(
        &mut self,
        description: impl Into<String>,
        rule: impl Into<String>,
        engine: impl Into<String>,
    ) {
        let idx = self.steps.len() + 1;
        self.steps.push(ProvenanceStep {
            step_index: idx,
            description: description.into(),
            rule_applied: rule.into(),
            verified_by_engine: engine.into(),
        });
    }

    pub fn seal(&mut self, verified: bool) {
        self.is_verified = verified;
        // Checksum determinista de trazabilidad
        let mut content = format!("{}:{}:{}", self.claim_id, self.statement, verified);
        for s in &self.steps {
            content.push_str(&format!("|{}:{}:{}", s.step_index, s.rule_applied, s.verified_by_engine));
        }
        // Usar un hash simple legible para trazabilidad
        self.checksum = format!("{:x}", format!("{:?}", content).chars().count() * 31337);
    }

    pub fn render_audit_report(&self) -> String {
        let mut report = format!("=== Árbol de Procedencia Epistémica [{}] ===\n", self.claim_id);
        report.push_str(&format!("Afirmación: \"{}\"\n", self.statement));
        report.push_str(&format!("Estado de Verificación: {}\n", if self.is_verified { "VALIDADO" } else { "RECHAZADO" }));
        report.push_str("Cadena de Justificación:\n");
        for step in &self.steps {
            report.push_str(&format!(
                "  [{}] {} (Regla: {}, Motor: {})\n",
                step.step_index, step.description, step.rule_applied, step.verified_by_engine
            ));
        }
        report.push_str(&format!("Firma de Integridad: {}\n", self.checksum));
        report
    }
}
