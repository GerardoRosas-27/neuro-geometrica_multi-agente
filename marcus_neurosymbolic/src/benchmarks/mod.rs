//! Módulo de Benchmarks de Falsificación de las Hipótesis de Gary Marcus
//! y de la Arquitectura FEP (Free Energy Principle & Surprise Reduction).
//!
//! Ejecuta los 9 benchmarks preregistrados:
//! - BM-1: Generalización Zero-Shot OOD en Variables (The Algebraic Mind, 2001)
//! - BM-2: Sistematicidad e Invarianza Composicional (The Algebraic Mind, 2001)
//! - BM-3: Física Núcleo y Permanencia Spelke (Rebooting AI, 2019)
//! - BM-4: Inferencia Causal frente a Correlación Espuria (Pearl-Marcus)
//! - BM-5: Vigilancia Epistémica y Supresión de Alucinaciones (Taming Silicon Valley, 2024)
//! - BM-6: Minimización de Energía Libre Variacional y Maximización de ELBO (FEP-Marcus H-FEP1)
//! - BM-7: Reducción Multimodal de Sorpresa Estructurada (FEP-Marcus H-FEP2)
//! - BM-8: Inferencia Activa y Selección Causal por G(pi) (FEP-Marcus H-FEP3)
//! - BM-9: Generalización OOD con Energía Libre Acotada (FEP-Marcus H-FEP4)

pub mod bm1_variable_binding;
pub mod bm2_compositionality;
pub mod bm3_spelke_physics;
pub mod bm4_causal_interventions;
pub mod bm5_anti_hallucination;
pub mod bm6_free_energy_minimization;
pub mod bm7_surprise_reduction;
pub mod bm8_active_inference_exploration;
pub mod bm9_ood_free_energy_generalization;

pub use bm1_variable_binding::{run_bm1_variable_binding, Bm1Results};
pub use bm2_compositionality::{run_bm2_compositionality, Bm2Results};
pub use bm3_spelke_physics::{run_bm3_spelke_physics, Bm3Results};
pub use bm4_causal_interventions::{run_bm4_causal_interventions, Bm4Results};
pub use bm5_anti_hallucination::{run_bm5_anti_hallucination, Bm5Results};
pub use bm6_free_energy_minimization::{run_bm6_free_energy_minimization, Bm6Results};
pub use bm7_surprise_reduction::{run_bm7_surprise_reduction, Bm7Results};
pub use bm8_active_inference_exploration::{run_bm8_active_inference_exploration, Bm8Results};
pub use bm9_ood_free_energy_generalization::{run_bm9_ood_free_energy_generalization, Bm9Results};

use serde::{Deserialize, Serialize};

/// Reporte consolidado de los 5 benchmarks originales de Marcus.
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
            md.push_str("Una o más hipótesis fallaron.\n");
        }
        md
    }
}

/// Reporte consolidado de la suite de Energía Libre y Reducción de Sorpresa (FEP-MARCUS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepBenchmarkReport {
    pub bm6: Bm6Results,
    pub bm7: Bm7Results,
    pub bm8: Bm8Results,
    pub bm9: Bm9Results,
    pub all_fep_passed: bool,
}

pub fn run_all_fep_benchmarks() -> FepBenchmarkReport {
    let bm6 = run_bm6_free_energy_minimization();
    let bm7 = run_bm7_surprise_reduction();
    let bm8 = run_bm8_active_inference_exploration();
    let bm9 = run_bm9_ood_free_energy_generalization();

    let all_fep_passed = bm6.passed_h_fep1 && bm7.passed_h_fep2 && bm8.passed_h_fep3 && bm9.passed_h_fep4;

    FepBenchmarkReport {
        bm6,
        bm7,
        bm8,
        bm9,
        all_fep_passed,
    }
}

impl FepBenchmarkReport {
    pub fn render_markdown_table(&self) -> String {
        let mut md = String::new();
        md.push_str("# Tabla de Resultados Consolidados: Benchmarks FEP-MARCUS\n");
        md.push_str("## Maximización de Energía Libre (ELBO) y Reducción de Sorpresa en Arquitectura Neuro-Simbólica\n\n");
        md.push_str("| Benchmark | Hipótesis FEP Evaluada | Métrica Línea Base No-FEP | Métrica FEP-MARCUS | Estado Preregistro |\n");
        md.push_str("|---|---|---|---|---|\n");

        md.push_str(&format!(
            "| **BM-6: Minimización F & ELBO** | H-FEP1: Convergencia de Energía Libre | Reducción = 0.0% (No optimizado) | **Reducción FE = {:.1}% (Monotonicidad = {:.1}%)** | {} |\n",
            self.bm6.free_energy_reduction_pct,
            self.bm6.monotonic_convergence_rate * 100.0,
            if self.bm6.passed_h_fep1 { "✅ **PASA H-FEP1 (CONFIRM)**" } else { "❌ FALLA" }
        ));

        md.push_str(&format!(
            "| **BM-7: Reducción de Sorpresa** | H-FEP2: Sorpresa Multimodal | Sorpresa No Resuelta = {:.2} | **Sorpresa Final = {:.2} (Reducción = {:.1}%)** | {} |\n",
            self.bm7.baseline_avg_unresolved_surprise,
            self.bm7.marcus_fep_final_surprise,
            self.bm7.surprise_reduction_pct,
            if self.bm7.passed_h_fep2 { "✅ **PASA H-FEP2 (CONFIRM)**" } else { "❌ FALLA" }
        ));

        md.push_str(&format!(
            "| **BM-8: Inferencia Activa G(pi)** | H-FEP3: Exploración Causal Epistémica | Exactitud Pasiva = {:.1}% | **Exactitud Activa = {:.1}% (Pref. Epistémica = {:.1}%)** | {} |\n",
            self.bm8.passive_agent_causal_accuracy * 100.0,
            self.bm8.active_inference_causal_accuracy * 100.0,
            self.bm8.epistemic_action_preference_rate * 100.0,
            if self.bm8.passed_h_fep3 { "✅ **PASA H-FEP3 (CONFIRM)**" } else { "❌ FALLA" }
        ));

        md.push_str(&format!(
            "| **BM-9: Generalización OOD** | H-FEP4: Cota de Energía Libre OOD | Exactitud OOD = {:.1}% (Sorpresa = {:.1}) | **Exactitud OOD = {:.1}% (FE = {:.2})** | {} |\n",
            self.bm9.statistical_baseline_ood_accuracy * 100.0,
            self.bm9.statistical_baseline_avg_surprise,
            self.bm9.marcus_fep_ood_accuracy * 100.0,
            self.bm9.marcus_fep_avg_free_energy,
            if self.bm9.passed_h_fep4 { "✅ **PASA H-FEP4 (CONFIRM)**" } else { "❌ FALLA" }
        ));

        md.push_str("\n> **Conclusión:** ");
        if self.all_fep_passed {
            md.push_str("Todas las hipótesis de la arquitectura basada 100% en maximizar la energía libre y reducir la sorpresa (H-FEP1, H-FEP2, H-FEP3, H-FEP4) quedan confirmadas empíricamente sin fallos. El sustrato cognitivo neuro-simbólico de Gary Marcus actúa como el andamiaje composicional idóneo para el Principio de Energía Libre de Karl Friston.\n");
        } else {
            md.push_str("Una o más hipótesis FEP no alcanzaron el umbral preregistrado.\n");
        }
        md
    }
}
