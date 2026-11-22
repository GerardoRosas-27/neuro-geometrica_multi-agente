//! Módulo de Ligamiento Dinámico de Variables y Operaciones Algebraicas.
//!
//! Basado en los principios de "The Algebraic Mind" (Gary Marcus, 2001, Caps. 2 y 3):
//! 1. Desacoplamiento estricto entre variables formales y valores instanciados.
//! 2. Ligamiento de Roles y Rellenadores (Role-Filler Binding) libre de diafonía (crosstalk).
//! 3. Soporte dual: simbólico explícito y Vector Symbolic Architecture (HRR / Smolensky-Marcus).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identificador de una variable formal (e.g., "x", "y", "agent", "patient").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarId(pub String);

impl VarId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// Valor que puede instanciar una variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VarValue {
    Atom(String),
    Number(f64),
    TokenId(String),
    Compound(Vec<VarValue>),
}

impl VarValue {
    pub fn as_atom(&self) -> Option<&str> {
        match self {
            VarValue::Atom(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Registro explícito de ligamiento de variables (Symbolic Variable Environment).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariableRegister {
    bindings: HashMap<VarId, VarValue>,
}

impl VariableRegister {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Liga una variable a un valor. Si ya estaba ligada al mismo valor, tiene éxito.
    /// Si estaba ligada a un valor diferente, devuelve un error de conflicto de ligamiento.
    pub fn bind(&mut self, var: VarId, val: VarValue) -> Result<(), String> {
        if let Some(existing) = self.bindings.get(&var) {
            if existing != &val {
                return Err(format!(
                    "Conflicto de ligamiento: variable '{}' ya ligada a {:?}, intento de ligar a {:?}",
                    var.0, existing, val
                ));
            }
        } else {
            self.bindings.insert(var, val);
        }
        Ok(())
    }

    pub fn get(&self, var: &VarId) -> Option<&VarValue> {
        self.bindings.get(var)
    }

    pub fn unbind(&mut self, var: &VarId) -> Option<VarValue> {
        self.bindings.remove(var)
    }

    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn all_bindings(&self) -> &HashMap<VarId, VarValue> {
        &self.bindings
    }
}

/// Implementación de Vector Symbolic Architecture (VSA / HRR - Holographic Reduced Representations)
/// para demostrar cómo una red neuronal o sustrato continuo puede implementar
/// ligamiento algebraico rol-rellenador sin diafonía (Smolensky, Plate, Marcus 2001 §3).
#[derive(Debug, Clone)]
pub struct HrrVector {
    pub dim: usize,
    pub values: Vec<f64>,
}

impl HrrVector {
    pub fn new_random(dim: usize, rng: &mut impl rand::Rng) -> Self {
        // Generador normal estándar mediante transformada Box-Muller
        let mut values = Vec::with_capacity(dim);
        let scale = 1.0 / (dim as f64).sqrt();
        while values.len() < dim {
            let u1: f64 = rng.gen::<f64>().max(1e-15);
            let u2: f64 = rng.gen::<f64>();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            values.push(z0 * scale);
            if values.len() < dim {
                let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
                values.push(z1 * scale);
            }
        }
        let mut vec = Self { dim, values };
        vec.normalize();
        vec
    }

    pub fn from_vec(values: Vec<f64>) -> Self {
        let dim = values.len();
        let mut vec = Self { dim, values };
        vec.normalize();
        vec
    }

    pub fn normalize(&mut self) {
        let norm: f64 = self.values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for v in &mut self.values {
                *v /= norm;
            }
        }
    }

    /// Convolución circular discreta: a ⊛ b
    pub fn circular_convolution(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim);
        let n = self.dim;
        let mut result = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let k = (i + n - j) % n;
                sum += self.values[j] * other.values[k];
            }
            result[i] = sum;
        }
        let mut out = Self { dim: n, values: result };
        out.normalize();
        out
    }

    /// Correlación circular (inverso aproximado para desatar / unbind): a* ⊛ c
    pub fn circular_correlation(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim);
        let n = self.dim;
        let mut result = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let k = (j + i) % n;
                sum += self.values[j] * other.values[k];
            }
            result[i] = sum;
        }
        let mut out = Self { dim: n, values: result };
        out.normalize();
        out
    }

    /// Superposición vectorial (adición): a + b
    pub fn superpose(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim);
        let result: Vec<f64> = self.values.iter().zip(other.values.iter()).map(|(a, b)| a + b).collect();
        let mut out = Self { dim: self.dim, values: result };
        out.normalize();
        out
    }

    /// Similitud coseno
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        assert_eq!(self.dim, other.dim);
        self.values.iter().zip(other.values.iter()).map(|(a, b)| a * b).sum()
    }
}

/// Patrón de regla algebraica universal (e.g. Gramática ABB vs ABA de Marcus et al., 1999).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlgebraicPattern {
    /// Tres elementos donde el 2do y 3ero son idénticos: x, y, y (ABB)
    ABB,
    /// Tres elementos donde el 1ero y 3ero son idénticos: x, y, x (ABA)
    ABA,
    /// Tres elementos donde el 1ero y 2do son idénticos: x, x, y (AAB)
    AAB,
    /// Identidad universal: f(x) = x
    Identity,
}

/// Evaluador de reglas algebraicas puras.
pub struct AlgebraicRuleEngine;

impl AlgebraicRuleEngine {
    /// Evalúa la regla de identidad f(x) = x para cualquier entrada fuera de distribución.
    pub fn eval_identity(input: &VarValue) -> VarValue {
        input.clone()
    }

    /// Reconoce el patrón gramatical algebraico (Marcus 1999) en una secuencia de 3 elementos,
    /// completamente independiente de si las palabras o fonemas fueron vistos antes.
    pub fn classify_three_token_grammar(seq: &[&str]) -> Option<AlgebraicPattern> {
        if seq.len() != 3 {
            return None;
        }
        let (a, b, c) = (seq[0], seq[1], seq[2]);
        if a != b && b == c {
            Some(AlgebraicPattern::ABB)
        } else if a == c && a != b {
            Some(AlgebraicPattern::ABA)
        } else if a == b && a != c {
            Some(AlgebraicPattern::AAB)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_variable_binding_and_conflict() {
        let mut reg = VariableRegister::new();
        let var_x = VarId::new("x");
        let val1 = VarValue::Atom("Dog".to_string());
        let val2 = VarValue::Atom("Cat".to_string());

        assert!(reg.bind(var_x.clone(), val1.clone()).is_ok());
        assert_eq!(reg.get(&var_x), Some(&val1));

        // Mismo valor es idempotente
        assert!(reg.bind(var_x.clone(), val1.clone()).is_ok());

        // Conflicto de ligamiento
        assert!(reg.bind(var_x.clone(), val2).is_err());
    }

    #[test]
    fn test_hrr_role_filler_binding_and_unbinding() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let dim = 512;

        let role_agent = HrrVector::new_random(dim, &mut rng);
        let role_patient = HrrVector::new_random(dim, &mut rng);

        let filler_john = HrrVector::new_random(dim, &mut rng);
        let filler_mary = HrrVector::new_random(dim, &mut rng);

        // Representación oracional: S = (Agent ⊛ John) + (Patient ⊛ Mary)
        let bound_agent = role_agent.circular_convolution(&filler_john);
        let bound_patient = role_patient.circular_convolution(&filler_mary);
        let sentence = bound_agent.superpose(&bound_patient);

        // Desatar el Agente usando correlation con role_agent
        let extracted_agent = role_agent.circular_correlation(&sentence);
        let sim_john = extracted_agent.cosine_similarity(&filler_john);
        let sim_mary = extracted_agent.cosine_similarity(&filler_mary);

        // John debe tener una similitud significativamente mayor que Mary al desatar Agent
        assert!(sim_john > 0.45, "sim_john: {}", sim_john);
        assert!(sim_mary < 0.20, "sim_mary: {}", sim_mary);
        assert!(sim_john > sim_mary + 0.25);
    }

    #[test]
    fn test_marcus_algebraic_grammar_ood() {
        // Fonemas artificiales OOD (nunca vistos)
        let tokens_abb = ["wo", "fe", "fe"];
        let tokens_aba = ["wo", "fe", "wo"];
        let tokens_aab = ["li", "li", "na"];

        assert_eq!(
            AlgebraicRuleEngine::classify_three_token_grammar(&tokens_abb),
            Some(AlgebraicPattern::ABB)
        );
        assert_eq!(
            AlgebraicRuleEngine::classify_three_token_grammar(&tokens_aba),
            Some(AlgebraicPattern::ABA)
        );
        assert_eq!(
            AlgebraicRuleEngine::classify_three_token_grammar(&tokens_aab),
            Some(AlgebraicPattern::AAB)
        );
    }
}
