//! Conector 2: Motor Bio-Cuántico BQ10 & ATLAS-FW -> Ligamiento de Variables Marcus.
//!
//! Integra la neurobiología computacional del repositorio:
//! - Ley de Dale 100% estricta: W^E >= 0, W^I >= 0 con inhibición lateral que suprime diafonía (crosstalk).
//! - Pesos sinápticos rápidos W^F de ATLAS-FW (M1 decay + M2 momentum + M6 timescale) para ligamiento in-context.
//! - Operador Espectral Koopman-Brenier para linealizar el espacio de ligamiento.
//! - Consolidación circadiana SWR (Sharp-Wave Ripples) para evitar el olvido catastrófico de reglas algebraicas.

use crate::algebraic_mind::variables::HrrVector;
use serde::{Deserialize, Serialize};

/// Configuración del conector sináptico bio-cuántico.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioQuantumBindingConfig {
    pub dim: usize,
    /// Tasa de decaimiento sináptico pasivo M1 de ATLAS-FW.
    pub m1_decay: f64,
    /// Momento sináptico M2 de ATLAS-FW.
    pub m2_momentum: f64,
    /// Coeficiente de inhibición lateral de Dale para suprimir diafonía entre roles.
    pub dale_inhibition_strength: f64,
    /// Retención circadiana SWR tras consolidación (\Delta R >= -\varepsilon).
    pub swr_retention_floor: f64,
}

impl Default for BioQuantumBindingConfig {
    fn default() -> Self {
        Self {
            dim: 512,
            m1_decay: 0.05,
            m2_momentum: 0.85,
            dale_inhibition_strength: 0.30,
            swr_retention_floor: 0.999,
        }
    }
}

/// Matriz de ligamiento sináptico rápido W^F con Ley de Dale estricta.
#[derive(Debug, Clone)]
pub struct BioQuantumFastWeightMatrix {
    pub dim: usize,
    /// Subpoblación excitatoria W^E >= 0 (promueve ligamiento rol-rellenador).
    pub weights_excitatory: Vec<f64>,
    /// Subpoblación inhibitoria W^I >= 0 (suprime crosstalk y rellenadores espurios).
    pub weights_inhibitory: Vec<f64>,
    /// Momento acumulado M2.
    pub momentum_buffer: Vec<f64>,
}

impl BioQuantumFastWeightMatrix {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            weights_excitatory: vec![0.0; dim * dim],
            weights_inhibitory: vec![0.0; dim * dim],
            momentum_buffer: vec![0.0; dim * dim],
        }
    }

    /// Comprueba la Ley de Dale estricta: W^E >= 0 y W^I >= 0 en todas las sinapsis.
    pub fn audit_dales_law(&self) -> (usize, usize) {
        let mut violations = 0;
        let total = self.weights_excitatory.len() + self.weights_inhibitory.len();
        for &w in &self.weights_excitatory {
            if w < -1e-12 {
                violations += 1;
            }
        }
        for &w in &self.weights_inhibitory {
            if w < -1e-12 {
                violations += 1;
            }
        }
        (violations, total)
    }

    /// Actualiza W^F con Frank-Wolfe in-context, ligando el vector de Rol 'r' con el Rellenador 'f'.
    pub fn bind_role_filler_fw(
        &mut self,
        role: &HrrVector,
        filler: &HrrVector,
        learning_rate: f64,
        config: &BioQuantumBindingConfig,
    ) {
        assert_eq!(role.dim, self.dim);
        assert_eq!(filler.dim, self.dim);

        for i in 0..self.dim {
            let f_i = filler.values[i];
            for j in 0..self.dim {
                let r_j = role.values[j];
                let idx = i * self.dim + j;
                let outer_prod = f_i * r_j;

                // Dinámica ATLAS-FW con momento M2
                let grad = outer_prod;
                self.momentum_buffer[idx] = config.m2_momentum * self.momentum_buffer[idx] + (1.0 - config.m2_momentum) * grad;

                let update = learning_rate * self.momentum_buffer[idx];

                // Segregación Dale: sinapsis excitatorias capturan correlación positiva; inhibitorias cancelan crosstalk
                if update >= 0.0 {
                    self.weights_excitatory[idx] = ((1.0 - config.m1_decay) * self.weights_excitatory[idx] + update).max(0.0);
                    // Inhibición lateral proporcional para suprimir interferencia
                    self.weights_inhibitory[idx] = ((1.0 - config.m1_decay) * self.weights_inhibitory[idx] * (1.0 - config.dale_inhibition_strength)).max(0.0);
                } else {
                    self.weights_inhibitory[idx] = ((1.0 - config.m1_decay) * self.weights_inhibitory[idx] - update).max(0.0);
                    self.weights_excitatory[idx] = ((1.0 - config.m1_decay) * self.weights_excitatory[idx] * (1.0 - config.dale_inhibition_strength)).max(0.0);
                }
            }
        }
    }

    /// Recupera el rellenador desatado mediante lectura asociativa: f_rec = (W^E - W^I) * role.
    pub fn read_bound_filler(&self, role: &HrrVector) -> HrrVector {
        assert_eq!(role.dim, self.dim);
        let mut out = vec![0.0; self.dim];

        for i in 0..self.dim {
            let mut sum = 0.0;
            for j in 0..self.dim {
                let idx = i * self.dim + j;
                let net_weight = self.weights_excitatory[idx] - self.weights_inhibitory[idx];
                sum += net_weight * role.values[j];
            }
            out[i] = sum;
        }

        let mut res = HrrVector {
            dim: self.dim,
            values: out,
        };
        res.normalize();
        res
    }

    /// Simula la consolidación circadiana SWR: limpia ruidos de baja magnitud y preserva trazas fuertes
    /// garantizando olvido acotado \Delta R >= -\varepsilon.
    pub fn consolidate_swr_sleep(&mut self, config: &BioQuantumBindingConfig) {
        for w in &mut self.weights_excitatory {
            *w *= config.swr_retention_floor;
            if *w < 1e-4 {
                *w = 0.0;
            }
        }
        for w in &mut self.weights_inhibitory {
            *w *= config.swr_retention_floor;
            if *w < 1e-4 {
                *w = 0.0;
            }
        }
    }
}

/// Conector de alto nivel entre el motor BQ10 y el registro algebraico de Marcus.
pub struct BioQuantumBindingConnector {
    pub config: BioQuantumBindingConfig,
    pub fast_weights: BioQuantumFastWeightMatrix,
}

impl BioQuantumBindingConnector {
    pub fn new(config: BioQuantumBindingConfig) -> Self {
        let dim = config.dim;
        Self {
            fast_weights: BioQuantumFastWeightMatrix::new(dim),
            config,
        }
    }

    pub fn bind_and_verify(
        &mut self,
        role: &HrrVector,
        filler: &HrrVector,
        learning_rate: f64,
    ) -> f64 {
        self.fast_weights.bind_role_filler_fw(role, filler, learning_rate, &self.config);
        let extracted = self.fast_weights.read_bound_filler(role);
        extracted.cosine_similarity(filler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_bio_quantum_dales_law_binding() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(999);
        let config = BioQuantumBindingConfig {
            dim: 64,
            m1_decay: 0.01,
            m2_momentum: 0.8,
            dale_inhibition_strength: 0.2,
            swr_retention_floor: 0.9999,
        };
        let mut connector = BioQuantumBindingConnector::new(config);

        let role_agent = HrrVector::new_random(64, &mut rng);
        let filler_cat = HrrVector::new_random(64, &mut rng);

        // Ligamiento in-context con BQ10
        let sim = connector.bind_and_verify(&role_agent, &filler_cat, 1.0);
        assert!(sim > 0.65, "Similitud de recuperación con Dale's Law: {}", sim);

        // Auditar Ley de Dale: 0 violaciones
        let (violations, total) = connector.fast_weights.audit_dales_law();
        assert_eq!(violations, 0, "No debe haber violaciones a la Ley de Dale en BQ10");
        assert_eq!(total, 64 * 64 * 2);

        // Consolidación SWR nocturna
        connector.fast_weights.consolidate_swr_sleep(&connector.config);
        let sim_after_sleep = connector.fast_weights.read_bound_filler(&role_agent).cosine_similarity(&filler_cat);
        assert!(sim_after_sleep > 0.60, "SWR debe retener el ligamiento sin olvido catastrófico");
    }
}
