//! Benchmark 2: Sistematicidad e Invarianza Composicional Jerárquica.
//!
//! Evalúa la hipótesis H2 de Gary Marcus (The Algebraic Mind, 2001, Cap. 4):
//! "The meaning of a composite structure is an asymmetric function of its syntactic tree.
//! Flat embeddings collapse Loves(John, Mary) and Loves(Mary, John); an algebraic architecture
//! guarantees structural asymmetry and systematic recombination without semantic loss."

use crate::algebraic_mind::syntax_tree::SyntaxNode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm2Results {
    pub name: String,
    pub total_trials: usize,
    pub bag_of_words_asymmetry_error: f64,
    pub marcus_engine_asymmetry_score: f64,
    pub recombination_accuracy: f64,
    pub passed_h2: bool,
}

pub fn run_bm2_compositionality() -> Bm2Results {
    let pairs = vec![
        ("John", "Mary", "Loves"),
        ("Cat", "Mouse", "Chases"),
        ("Box", "Table", "On"),
        ("Water", "Glass", "In"),
        ("Teacher", "Student", "Instructs"),
    ];

    let mut bow_symmetry_collisions = 0;
    let mut marcus_correct_asymmetries = 0;

    for (a, b, pred) in &pairs {
        let node_ab = SyntaxNode::pred(*pred, vec![SyntaxNode::atom(*a), SyntaxNode::atom(*b)]);
        let node_ba = SyntaxNode::pred(*pred, vec![SyntaxNode::atom(*b), SyntaxNode::atom(*a)]);

        // 1. Simulación Bolsa de Palabras (BoW / Unigram Flat Model):
        // BoW([A, Pred, B]) == BoW([B, Pred, A]) -> Colisión de simetría 100%
        let bow_ab = vec![*a, *pred, *b];
        let mut bow_ab_sorted = bow_ab.clone();
        bow_ab_sorted.sort();

        let bow_ba = vec![*b, *pred, *a];
        let mut bow_ba_sorted = bow_ba.clone();
        bow_ba_sorted.sort();

        if bow_ab_sorted == bow_ba_sorted {
            bow_symmetry_collisions += 1;
        }

        // 2. Modelo MARCUS-I:
        // Preserva el orden sintáctico estricto en el árbol
        if node_ab != node_ba && !node_ab.exact_structural_match(&node_ba) {
            marcus_correct_asymmetries += 1;
        }
    }

    // Evaluación de recombinación composicional en 50 combinaciones sistemáticas
    let mut recombination_hits = 0;
    let total_recombinations = 50;

    let predicates = ["Inside", "OnTopOf", "SupportedBy", "Behind", "AdjacentTo"];
    let entities = ["Sphere", "Cube", "Cylinder", "Pyramid", "Torus"];

    for i in 0..total_recombinations {
        let p = predicates[i % predicates.len()];
        let e1 = entities[i % entities.len()];
        let e2 = entities[(i + 1) % entities.len()];

        let tree = SyntaxNode::pred(p, vec![SyntaxNode::atom(e1), SyntaxNode::atom(e2)]);
        // Sustituir e1 por una variable x y comprobar consistencia
        let templated = SyntaxNode::pred(p, vec![SyntaxNode::var("x"), SyntaxNode::atom(e2)]);
        let restored = templated.substitute("x", &SyntaxNode::atom(e1));

        if restored == tree && restored.tree_depth() == 2 {
            recombination_hits += 1;
        }
    }

    let marcus_asym_score = marcus_correct_asymmetries as f64 / pairs.len() as f64;
    let bow_asym_error = bow_symmetry_collisions as f64 / pairs.len() as f64;
    let recomb_acc = recombination_hits as f64 / total_recombinations as f64;

    Bm2Results {
        name: "BM-2: Sistematicidad e Invarianza Composicional".to_string(),
        total_trials: pairs.len() + total_recombinations,
        bag_of_words_asymmetry_error: bow_asym_error,
        marcus_engine_asymmetry_score: marcus_asym_score,
        recombination_accuracy: recomb_acc,
        passed_h2: marcus_asym_score == 1.0 && recomb_acc == 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm2_execution() {
        let res = run_bm2_compositionality();
        assert!(res.passed_h2);
        assert_eq!(res.marcus_engine_asymmetry_score, 1.0);
        assert_eq!(res.recombination_accuracy, 1.0);
        assert_eq!(res.bag_of_words_asymmetry_error, 1.0);
    }
}
