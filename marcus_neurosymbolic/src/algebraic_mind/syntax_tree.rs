//! Módulo de Sintaxis Jerárquica y Composicionalidad Recursiva.
//!
//! Basado en Gary Marcus (The Algebraic Mind, Cap. 4):
//! "The mind requires structured representations such that the meaning of a whole
//! is a systematic function of the parts and the syntactic rules used to combine them."

use serde::{Deserialize, Serialize};

/// Expresión composicional en forma de árbol sintáctico abstracto.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SyntaxNode {
    /// Átomo constante (símbolo o valor elemental).
    Atom(String),
    /// Variable libre o ligada.
    Variable(String),
    /// Predicado n-ario con lista ordenada de argumentos.
    /// Crucial para Marcus: Predicate("Loves", [A, B]) != Predicate("Loves", [B, A]).
    Predicate {
        name: String,
        arguments: Vec<SyntaxNode>,
    },
    /// Conjunción lógica (Y).
    And(Vec<SyntaxNode>),
    /// Disyunción lógica (O).
    Or(Vec<SyntaxNode>),
    /// Negación lógica (NO).
    Not(Box<SyntaxNode>),
    /// Implicación lógica: Antecedente -> Consecuente.
    Implies {
        antecedent: Box<SyntaxNode>,
        consequent: Box<SyntaxNode>,
    },
}

impl SyntaxNode {
    pub fn atom(s: impl Into<String>) -> Self {
        Self::Atom(s.into())
    }

    pub fn var(s: impl Into<String>) -> Self {
        Self::Variable(s.into())
    }

    pub fn pred(name: impl Into<String>, args: Vec<SyntaxNode>) -> Self {
        Self::Predicate {
            name: name.into(),
            arguments: args,
        }
    }

    /// Comprueba si esta estructura es idéntica a otra preservando el orden estricto de argumentos.
    pub fn exact_structural_match(&self, other: &Self) -> bool {
        self == other
    }

    /// Retorna la profundidad jerárquica del árbol sintáctico.
    pub fn tree_depth(&self) -> usize {
        match self {
            Self::Atom(_) | Self::Variable(_) => 1,
            Self::Predicate { arguments, .. } => {
                1 + arguments.iter().map(|a| a.tree_depth()).max().unwrap_or(0)
            }
            Self::And(nodes) | Self::Or(nodes) => {
                1 + nodes.iter().map(|n| n.tree_depth()).max().unwrap_or(0)
            }
            Self::Not(inner) => 1 + inner.tree_depth(),
            Self::Implies { antecedent, consequent } => {
                1 + antecedent.tree_depth().max(consequent.tree_depth())
            }
        }
    }

    /// Sustituye una variable libre por una subexpresión o átomo.
    pub fn substitute(&self, var_name: &str, replacement: &SyntaxNode) -> Self {
        match self {
            Self::Atom(_) => self.clone(),
            Self::Variable(v) => {
                if v == var_name {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            Self::Predicate { name, arguments } => Self::Predicate {
                name: name.clone(),
                arguments: arguments
                    .iter()
                    .map(|arg| arg.substitute(var_name, replacement))
                    .collect(),
            },
            Self::And(nodes) => Self::And(
                nodes.iter().map(|n| n.substitute(var_name, replacement)).collect(),
            ),
            Self::Or(nodes) => Self::Or(
                nodes.iter().map(|n| n.substitute(var_name, replacement)).collect(),
            ),
            Self::Not(inner) => Self::Not(Box::new(inner.substitute(var_name, replacement))),
            Self::Implies { antecedent, consequent } => Self::Implies {
                antecedent: Box::new(antecedent.substitute(var_name, replacement)),
                consequent: Box::new(consequent.substitute(var_name, replacement)),
            },
        }
    }

    /// Formatea la expresión en notación lógica estándar con paréntesis explícitos.
    pub fn to_pretty_string(&self) -> String {
        match self {
            Self::Atom(s) => s.clone(),
            Self::Variable(v) => format!("?{}", v),
            Self::Predicate { name, arguments } => {
                let args_str: Vec<String> = arguments.iter().map(|a| a.to_pretty_string()).collect();
                format!("{}({})", name, args_str.join(", "))
            }
            Self::And(nodes) => {
                let sub: Vec<String> = nodes.iter().map(|n| n.to_pretty_string()).collect();
                format!("({})", sub.join(" ∧ "))
            }
            Self::Or(nodes) => {
                let sub: Vec<String> = nodes.iter().map(|n| n.to_pretty_string()).collect();
                format!("({})", sub.join(" ∨ "))
            }
            Self::Not(inner) => format!("¬{}", inner.to_pretty_string()),
            Self::Implies { antecedent, consequent } => {
                format!("({} → {})", antecedent.to_pretty_string(), consequent.to_pretty_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asymmetric_compositionality() {
        // En Marcus (2001), Loves(John, Mary) != Loves(Mary, John).
        // Las bolsas de palabras (bag of words) o modelos no estructurados colapsan ambos.
        let john = SyntaxNode::atom("John");
        let mary = SyntaxNode::atom("Mary");

        let john_loves_mary = SyntaxNode::pred("Loves", vec![john.clone(), mary.clone()]);
        let mary_loves_john = SyntaxNode::pred("Loves", vec![mary.clone(), john.clone()]);

        assert_ne!(john_loves_mary, mary_loves_john);
        assert!(!john_loves_mary.exact_structural_match(&mary_loves_john));
        assert_eq!(john_loves_mary.to_pretty_string(), "Loves(John, Mary)");
        assert_eq!(mary_loves_john.to_pretty_string(), "Loves(Mary, John)");
    }

    #[test]
    fn test_variable_substitution_in_hierarchy() {
        // Rule: On(?x, Table) ∧ In(Table, Room)
        let pattern = SyntaxNode::And(vec![
            SyntaxNode::pred("On", vec![SyntaxNode::var("x"), SyntaxNode::atom("Table")]),
            SyntaxNode::pred("In", vec![SyntaxNode::atom("Table"), SyntaxNode::atom("Room")]),
        ]);

        let instantiated = pattern.substitute("x", &SyntaxNode::atom("Book"));
        let expected = SyntaxNode::And(vec![
            SyntaxNode::pred("On", vec![SyntaxNode::atom("Book"), SyntaxNode::atom("Table")]),
            SyntaxNode::pred("In", vec![SyntaxNode::atom("Table"), SyntaxNode::atom("Room")]),
        ]);

        assert_eq!(instantiated, expected);
        assert_eq!(instantiated.tree_depth(), 3);
        assert_eq!(instantiated.to_pretty_string(), "(On(Book, Table) ∧ In(Table, Room))");
    }
}
