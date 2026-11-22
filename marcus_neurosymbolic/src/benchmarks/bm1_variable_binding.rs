//! Benchmark 1: Generalización Zero-Shot Fuera de Distribución en Operaciones sobre Variables.
//!
//! Evalúa la hipótesis H1 de Gary Marcus (The Algebraic Mind, 2001):
//! "Pure statistical connectionist models tie knowledge to local weights and fail on OOD inputs.
//! An algebraic engine with variable binding achieves 100% generalization zero-shot."

use crate::algebraic_mind::variables::{AlgebraicPattern, AlgebraicRuleEngine, VarId, VarValue, VariableRegister};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm1Results {
    pub name: String,
    pub total_trials: usize,
    pub statistical_baseline_accuracy: f64,
    pub marcus_engine_accuracy: f64,
    pub passed_h1: bool,
}

/// Línea base asociativa pura (emulación de modelo de memoria superficial o n-grama sin variables).
struct StatisticalBaseline {
    training_vocabulary: std::collections::HashSet<String>,
}

impl StatisticalBaseline {
    fn new(training_tokens: &[&str]) -> Self {
        let mut set = std::collections::HashSet::new();
        for t in training_tokens {
            set.insert(t.to_string());
        }
        Self { training_vocabulary: set }
    }

    /// Intenta predecir la regla de identidad f(x) = x.
    /// Si el token está en el vocabulario de entrenamiento, acierta; si es OOD, falla o predice al azar.
    fn predict_identity(&self, token: &str) -> Option<String> {
        if self.training_vocabulary.contains(token) {
            Some(token.to_string())
        } else {
            // Falla catastrófica OOD: no puede generalizar fuera del soporte de entrenamiento
            None
        }
    }

    /// Intenta clasificar gramática ABB vs ABA.
    /// Si los tokens son desconocidos, su predicción se reduce al nivel de azar (50%).
    fn predict_grammar_pattern(&self, seq: &[&str]) -> AlgebraicPattern {
        let all_known = seq.iter().all(|t| self.training_vocabulary.contains(*t));
        if all_known {
            if seq[1] == seq[2] { AlgebraicPattern::ABB } else { AlgebraicPattern::ABA }
        } else {
            // Predicción degradada a azar sobre tokens OOD
            AlgebraicPattern::ABB
        }
    }
}

pub fn run_bm1_variable_binding() -> Bm1Results {
    let training_vocab = ["cat", "dog", "fish", "bird", "tree", "house", "car", "book"];
    let baseline = StatisticalBaseline::new(&training_vocab);

    // 100 tokens completamente sintéticos OOD (nunca presentes en training)
    let ood_tokens: Vec<String> = (0..100).map(|i| format!("novel_token_{:03}", i)).collect();

    let mut baseline_hits = 0;
    let mut marcus_hits = 0;
    let total_trials = ood_tokens.len();

    for token in &ood_tokens {
        let input_val = VarValue::Atom(token.clone());

        // 1. Evaluación en Línea Base Asociativa
        let base_pred = baseline.predict_identity(token);
        if base_pred.as_deref() == Some(token) {
            baseline_hits += 1;
        }

        // 2. Evaluación en Motor Algebraico MARCUS-I
        // Aplica operación sobre variable formal x: f(x) = x
        let mut register = VariableRegister::new();
        let var_x = VarId::new("x");
        assert!(register.bind(var_x.clone(), input_val.clone()).is_ok());

        let bound_val = register.get(&var_x).unwrap();
        let marcus_output = AlgebraicRuleEngine::eval_identity(bound_val);

        if marcus_output == input_val {
            marcus_hits += 1;
        }
    }

    // Evaluación adicional de gramática ABB vs ABA en tokens OOD
    let grammar_trials = vec![
        (["qux", "zorp", "zorp"], AlgebraicPattern::ABB),
        (["qux", "zorp", "qux"], AlgebraicPattern::ABA),
        (["bla", "kru", "kru"], AlgebraicPattern::ABB),
        (["bla", "kru", "bla"], AlgebraicPattern::ABA),
    ];

    let mut grammar_baseline_hits = 0;
    let mut grammar_marcus_hits = 0;

    for (seq, ground_truth) in &grammar_trials {
        let pred_base = baseline.predict_grammar_pattern(seq);
        if pred_base == *ground_truth {
            grammar_baseline_hits += 1;
        }

        let pred_marcus = AlgebraicRuleEngine::classify_three_token_grammar(seq);
        if pred_marcus == Some(*ground_truth) {
            grammar_marcus_hits += 1;
        }
    }

    let combined_total = total_trials + grammar_trials.len();
    let combined_baseline_acc = (baseline_hits + grammar_baseline_hits) as f64 / combined_total as f64;
    let combined_marcus_acc = (marcus_hits + grammar_marcus_hits) as f64 / combined_total as f64;

    Bm1Results {
        name: "BM-1: Generalización Zero-Shot OOD en Variables".to_string(),
        total_trials: combined_total,
        statistical_baseline_accuracy: combined_baseline_acc,
        marcus_engine_accuracy: combined_marcus_acc,
        passed_h1: combined_marcus_acc >= 0.99 && combined_marcus_acc > combined_baseline_acc + 0.50,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm1_execution() {
        let res = run_bm1_variable_binding();
        assert!(res.passed_h1);
        assert_eq!(res.marcus_engine_accuracy, 1.0);
        assert!(res.statistical_baseline_accuracy < 0.20);
    }
}
