//! Módulo de Benchmarks de Falsificación de las Hipótesis de Gary Marcus.
//!
//! Ejecuta los 5 benchmarks preregistrados:
//! - BM-1: Generalización Zero-Shot OOD en Variables (The Algebraic Mind, 2001)
//! - BM-2: Sistematicidad e Invarianza Composicional (The Algebraic Mind, 2001)
//! - BM-3: Física Núcleo y Permanencia Spelke (Rebooting AI, 2019)
//! - BM-4: Inferencia Causal frente a Correlación Espuria (Pearl-Marcus)
//! - BM-5: Vigilancia Epistémica y Supresión de Alucinaciones (Taming Silicon Valley, 2024)

pub mod bm1_variable_binding;
pub mod bm2_compositionality;
pub mod bm3_spelke_physics;
pub mod bm4_causal_interventions;
pub mod bm5_anti_hallucination;

pub use bm1_variable_binding::{run_bm1_variable_binding, Bm1Results};
pub use bm2_compositionality::{run_bm2_compositionality, Bm2Results};
pub use bm3_spelke_physics::{run_bm3_spelke_physics, Bm3Results};
pub use bm4_causal_interventions::{run_bm4_causal_interventions, Bm4Results};
pub use bm5_anti_hallucination::{run_bm5_anti_hallucination, Bm5Results};

use serde::{Deserialize, Serialize};

/// Reporte consolidado de los 5 benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub bm1: Bm1Results,
    pub bm2: Bm2Results,
    pub bm3: Bm3Results,
    pub bm4: Bm4Results,
    pub bm5: Bm5Results,
    pub all_passed: bool,
}

pub fn run_all_benchmarks() -> BenchmarkReport {
    let bm1 = run_bm1_variable_binding();
    let bm2 = run_bm2_compositionality();
    let bm3 = run_bm3_spelke_physics();
    let bm4 = run_bm4_causal_interventions();
    let bm5 = run_bm5_anti_hallucination();

    let all_passed = bm1.passed_h1 && bm2.passed_h2 && bm3.passed_h3 && bm4.passed_h4 && bm5.passed_h5;

    BenchmarkReport {
        bm1,
        bm2,
        bm3,
        bm4,
        bm5,
        all_passed,
    }
}

impl BenchmarkReport {
    pub fn render_markdown_table(&self) -> String {
        let mut md = String::new();
        md.push_str("# Tabla de Resultados Consolidados: Benchmarks MARCUS-I\n\n");
        md.push_str("| Benchmark | Hipótesis Evaluada | Métrica Predictor Asociativo | Métrica MARCUS-I | Estado Preregistro |\n");
        md.push_str("|---|---|---|---|---|\n");

        md.push_str(&format!(
            "| **BM-1: Variables OOD** | H1: Ligamiento de Variables | Exactitud OOD = {:.1}% | **Exactitud OOD = {:.1}%** | {} |\n",
            self.bm1.statistical_baseline_accuracy * 100.0,
            self.bm1.marcus_engine_accuracy * 100.0,
            if self.bm1.passed_h1 { "✅ **PASA H1 (CONFIRM)**" } else { "❌ FALLA" }
        ));

        md.push_str(&format!(
            "| **BM-2: Composicionalidad** | H2: Jerarquía no conmutable | Error Asimetría = {:.1}% | **Asimetría = {:.1}% (Recomb = {:.1}%)** | {} |\n",
            self.bm2.bag_of_words_asymmetry_error * 100.0,
            self.bm2.marcus_engine_asymmetry_score * 100.0,
            self.bm2.recombination_accuracy * 100.0,
            if self.bm2.passed_h2 { "✅ **PASA H2 (CONFIRM)**" } else { "❌ FALLA" }
        ));

        md.push_str(&format!(
            "| **BM-3: Spelke Physics** | H3: Permanencia y Solidez | FPR = N/A | **Recall Violación = {:.1}% (FPR = {:.1}%)** | {} |\n",
            self.bm3.violation_recall * 100.0,
            self.bm3.false_positive_rate * 100.0,
            if self.bm3.passed_h3 { "✅ **PASA H3 (CONFIRM)**" } else { "❌ FALLA" }
        ));

        md.push_str(&format!(
            "| **BM-4: Inferencia Causal** | H4: Pearl Intervención do(x) | Error Causal = {:.2} | **Error Causal = {:.4}** | {} |\n",
            self.bm4.observational_correlation_error,
            self.bm4.marcus_causal_intervention_error,
            if self.bm4.passed_h4 { "✅ **PASA H4 (CONFIRM)**" } else { "❌ FALLA" }
        ));

        md.push_str(&format!(
            "| **BM-5: Anti-Alucinación** | H5: Vigilancia Epistémica | Tasa Alucinación = {:.1}% | **Tasa Alucinación = {:.1}% (Traza = {:.1}%)** | {} |\n",
            self.bm5.associative_baseline_hallucination_rate * 100.0,
            self.bm5.marcus_hallucination_admittance_rate * 100.0,
            self.bm5.provenance_traceability_rate * 100.0,
            if self.bm5.passed_h5 { "✅ **PASA H5 (CONFIRM)**" } else { "❌ FALLA" }
        ));

        md.push_str("\n> **Conclusión:** ");
        if self.all_passed {
            md.push_str("Todas las hipótesis preregistradas (H1, H2, H3, H4, H5) quedan confirmadas empíricamente sin fallos. La arquitectura neuro-simbólica MARCUS-I supera de manera categórica las limitaciones epistemológicas del conexionismo asociativo puro.\n");
        } else {
            md.push_str("Una o más hipótesis fueron falsificadas según los criterios preregistrados.\n");
        }

        md
    }
}
