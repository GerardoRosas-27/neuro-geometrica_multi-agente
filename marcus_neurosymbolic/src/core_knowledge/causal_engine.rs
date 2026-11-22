//! Módulo de Inferencia Causal Intervencionista (Pearl-Marcus Causal Engine).
//!
//! Basado en Gary Marcus (Rebooting AI, Cap. 3 y The Next Decade in AI, 2020):
//! "True intelligence requires causal understanding: knowing not merely that two events
//! correlate in data, but what will happen if we actively intervene on the world using do(x)."

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Variable causal en un Grafo Causal Dirigido (DAG).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CausalVar(pub String);

impl CausalVar {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// Función de asignación estructural para un Modelo Causal Estructural (SCM).
/// X_i = f(Parents(X_i)) + Noise
#[derive(Debug, Clone)]
pub struct StructuralEquation {
    /// Coeficientes lineales de los padres: var -> peso.
    pub parent_weights: HashMap<CausalVar, f64>,
    /// Sesgo o intercepto base.
    pub bias: f64,
    /// Desviación típica del ruido exógeno U_i.
    pub noise_std: f64,
}

impl StructuralEquation {
    pub fn new(bias: f64, noise_std: f64) -> Self {
        Self {
            parent_weights: HashMap::new(),
            bias,
            noise_std,
        }
    }

    pub fn with_parent(mut self, parent: CausalVar, weight: f64) -> Self {
        self.parent_weights.insert(parent, weight);
        self
    }

    pub fn compute(&self, parent_values: &HashMap<CausalVar, f64>, exogenous_noise: f64) -> f64 {
        let mut val = self.bias + exogenous_noise;
        for (parent, weight) in &self.parent_weights {
            if let Some(&p_val) = parent_values.get(parent) {
                val += weight * p_val;
            }
        }
        val
    }
}

/// Modelo Causal Estructural (SCM) con soporte de intervenciones do(X = x).
#[derive(Debug, Clone)]
pub struct CausalEngine {
    /// Ecuaciones estructurales originales para cada variable endógena.
    equations: HashMap<CausalVar, StructuralEquation>,
    /// Orden topológico de las variables en el DAG.
    topological_order: Vec<CausalVar>,
    /// Intervenciones activas: do(Var = val). Reemplazan la ecuación estructural mutilando aristas entrantes.
    active_interventions: HashMap<CausalVar, f64>,
}

impl CausalEngine {
    pub fn new() -> Self {
        Self {
            equations: HashMap::new(),
            topological_order: Vec::new(),
            active_interventions: HashMap::new(),
        }
    }

    /// Añade una variable y su ecuación estructural, especificando el orden de dependencia causal.
    pub fn add_structural_equation(&mut self, var: CausalVar, eq: StructuralEquation) {
        self.equations.insert(var.clone(), eq);
        if !self.topological_order.contains(&var) {
            self.topological_order.push(var);
        }
    }

    /// Aplica el operador do(X = value) de Judea Pearl.
    /// Mutila el grafo eliminando todas las dependencias entrantes de X y fijando su valor a una constante.
    pub fn set_do_intervention(&mut self, var: CausalVar, value: f64) {
        self.active_interventions.insert(var, value);
    }

    /// Limpia todas las intervenciones activas, restaurando el grafo observacional natural.
    pub fn clear_interventions(&mut self) {
        self.active_interventions.clear();
    }

    /// Comprueba si la variable está registrada en el grafo causal.
    pub fn contains_var(&self, var: &CausalVar) -> bool {
        self.equations.contains_key(var)
    }

    /// Retorna el número de padres causales directos de la variable.
    pub fn parent_count(&self, var: &CausalVar) -> usize {
        self.equations.get(var).map_or(0, |eq| eq.parent_weights.len())
    }

    /// Simula un forward pass del modelo causal (con o sin intervenciones activas).
    /// Si `noise_samples` es provisto, usa esos valores; de lo contrario, asume ruido esperado = 0.
    pub fn simulate(&self, noise_samples: Option<&HashMap<CausalVar, f64>>) -> HashMap<CausalVar, f64> {
        let mut values = HashMap::new();

        for var in &self.topological_order {
            // Si la variable está intervenida con do(X = x), su valor es fijado directamente
            if let Some(&intervened_val) = self.active_interventions.get(var) {
                values.insert(var.clone(), intervened_val);
            } else if let Some(eq) = self.equations.get(var) {
                let noise = noise_samples
                    .and_then(|ns| ns.get(var).copied())
                    .unwrap_or(0.0);
                let val = eq.compute(&values, noise);
                values.insert(var.clone(), val);
            }
        }

        values
    }

    /// Estima el efecto causal esperado E[Target | do(Intervened = val)].
    pub fn estimate_interventional_expectation(
        &mut self,
        intervened: CausalVar,
        intervened_val: f64,
        target: &CausalVar,
    ) -> f64 {
        self.set_do_intervention(intervened.clone(), intervened_val);
        let sample = self.simulate(None);
        self.active_interventions.remove(&intervened);
        sample.get(target).copied().unwrap_or(0.0)
    }

    /// Computa la correlación puramente observacional (lo que un modelo asociativo o LLM captura):
    /// En presencia de un confusor común Z -> X y Z -> Y, X e Y correlacionan fuertemente,
    /// pero do(X) no tiene efecto sobre Y si no existe arista X -> Y.
    pub fn is_causally_dependent(&mut self, cause: &CausalVar, effect: &CausalVar) -> bool {
        // Test de intervención de Marcus-Pearl: ¿Varía E[Y | do(X = x1)] frente a E[Y | do(X = x2)]?
        let y_val1 = self.estimate_interventional_expectation(cause.clone(), 0.0, effect);
        let y_val2 = self.estimate_interventional_expectation(cause.clone(), 10.0, effect);
        (y_val2 - y_val1).abs() > 1e-5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spurious_correlation_vs_causal_intervention() {
        // Caso clásico de Marcus:
        // Confusor Z = "Temporada de Invierno"
        // X = "Gente usando abrigos" (Z -> X)
        // Y = "Casos de Gripe"       (Z -> Y)
        // Nota: ¡Los abrigos (X) NO causan la gripe (Y)! No hay enlace X -> Y.
        let mut engine = CausalEngine::new();

        let z = CausalVar::new("WinterSeason");
        let x = CausalVar::new("WearingCoats");
        let y = CausalVar::new("FluCases");

        // Z es exógena: base = 1.0 (invierno)
        engine.add_structural_equation(z.clone(), StructuralEquation::new(1.0, 0.0));

        // X depende de Z: Coats = 2.0 * Winter + 0.1
        let eq_x = StructuralEquation::new(0.1, 0.0).with_parent(z.clone(), 2.0);
        engine.add_structural_equation(x.clone(), eq_x);

        // Y depende de Z: Flu = 5.0 * Winter + 0.2 (SIN DEPENDENCIA DE X)
        let eq_y = StructuralEquation::new(0.2, 0.0).with_parent(z.clone(), 5.0);
        engine.add_structural_equation(y.clone(), eq_y);

        // 1. Simulación observacional: ambos son altos porque Z es alto
        let obs = engine.simulate(None);
        assert!(*obs.get(&x).unwrap() > 2.0);
        assert!(*obs.get(&y).unwrap() > 5.0);

        // 2. Comprobación intervencionista: ¿Causa X a Y?
        // Si forzamos do(WearingCoats = 100.0), ¿cambian los casos de gripe Y?
        let flu_with_no_coats = engine.estimate_interventional_expectation(x.clone(), 0.0, &y);
        let flu_with_100_coats = engine.estimate_interventional_expectation(x.clone(), 100.0, &y);

        assert_eq!(flu_with_no_coats, flu_with_100_coats);
        assert!(!engine.is_causally_dependent(&x, &y), "Coats no deben causar Flu!");

        // 3. En cambio Z sí causa Y:
        assert!(engine.is_causally_dependent(&z, &y), "Winter sí causa Flu!");
    }
}
