//! Tensor network definida directamente sobre el multigrafo pyrochlore.
//!
//! Cada enlace físico recibe un índice virtual independiente. Esto elimina la
//! noción de "enlace no local" de la grilla PEPS, a costa de una contracción
//! genérica cuyo coste depende del treewidth del grafo.

use crate::quantum_spin_thermodynamic_engine::QuantumSpinBond;
use oxicuda_tn::contraction::einsum::LabelledTensor;
use oxicuda_tn::contraction::path::{execute_path, greedy_path};
use oxicuda_tn::error::TnResult;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphTensorNetworkReport {
    pub sites: usize,
    pub represented_bonds: usize,
    pub tensor_count: usize,
    pub contraction_steps: usize,
    pub contraction_cost: usize,
    pub state_dimension: usize,
    pub nonzero_amplitudes: usize,
    pub norm: f64,
    pub single_spin_entropy: f64,
    pub heisenberg_energy: f64,
    pub optimization_available: bool,
}

#[derive(Clone, Debug)]
pub struct PyrochloreGraphTensorNetwork {
    pub site_tensors: Vec<LabelledTensor>,
    pub bonds: Vec<QuantumSpinBond>,
    pub virtual_bond_count: usize,
    pub sites: usize,
}

impl PyrochloreGraphTensorNetwork {
    /// Estado GHZ normalizado como primera prueba de una red tensorial en el
    /// grafo real. El copy tensor de cada sitio conecta todos sus índices
    /// virtuales y el índice físico.
    pub fn ghz(sites: usize, bonds: &[QuantumSpinBond]) -> TnResult<Self> {
        let expanded = expand_bonds(bonds);
        let mut incident = vec![Vec::<usize>::new(); sites];
        for (virtual_bond, &(a, b)) in expanded.iter().enumerate() {
            incident[a].push(virtual_bond);
            incident[b].push(virtual_bond);
        }
        let mut site_tensors = Vec::with_capacity(sites);
        for site in 0..sites {
            incident[site].sort_unstable();
            let mut labels = incident[site]
                .iter()
                .map(|bond| label(*bond))
                .collect::<Vec<_>>();
            labels.push(label(expanded.len() + site));
            let dims = vec![2; labels.len()];
            let mut data = vec![0.0; 1usize << labels.len()];
            data[0] = if site == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let last = data.len() - 1;
            data[last] = if site == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            site_tensors.push(LabelledTensor::new(data, dims, labels)?);
        }
        Ok(Self {
            site_tensors,
            bonds: bonds.to_vec(),
            virtual_bond_count: expanded.len(),
            sites,
        })
    }

    pub fn contract_state(&self) -> TnResult<(Vec<f64>, usize, usize)> {
        let path = greedy_path(&self.site_tensors)?;
        let contracted = execute_path(self.site_tensors.clone(), &path)?;
        let mut state = vec![0.0; 1usize << self.sites];
        for flat in 0..contracted.data.len() {
            let mut remainder = flat;
            let mut axis_values = vec![0usize; contracted.dims.len()];
            for axis in (0..contracted.dims.len()).rev() {
                axis_values[axis] = remainder % contracted.dims[axis];
                remainder /= contracted.dims[axis];
            }
            let mut basis = 0usize;
            for (axis, tensor_label) in contracted.labels.iter().enumerate() {
                let label_index = label_index(*tensor_label);
                if label_index >= self.virtual_bond_count {
                    let site = label_index - self.virtual_bond_count;
                    basis |= axis_values[axis] << site;
                }
            }
            state[basis] = contracted.data[flat];
        }
        Ok((state, path.steps.len(), path.total_cost))
    }

    pub fn report(&self, exchange: f64) -> TnResult<GraphTensorNetworkReport> {
        let (state, contraction_steps, contraction_cost) = self.contract_state()?;
        let norm = state
            .iter()
            .map(|amplitude| amplitude * amplitude)
            .sum::<f64>()
            .sqrt();
        let single_spin_entropy = single_spin_entropy(&state, 0);
        let energy = heisenberg_energy(&state, &self.bonds, exchange);
        Ok(GraphTensorNetworkReport {
            sites: self.sites,
            represented_bonds: self.virtual_bond_count,
            tensor_count: self.site_tensors.len(),
            contraction_steps,
            contraction_cost,
            state_dimension: state.len(),
            nonzero_amplitudes: state
                .iter()
                .filter(|amplitude| amplitude.abs() > 1.0e-12)
                .count(),
            norm,
            single_spin_entropy,
            heisenberg_energy: energy,
            optimization_available: false,
        })
    }
}

fn expand_bonds(bonds: &[QuantumSpinBond]) -> Vec<(usize, usize)> {
    bonds
        .iter()
        .flat_map(|bond| std::iter::repeat_n((bond.a, bond.b), bond.multiplicity as usize))
        .collect()
}

fn label(index: usize) -> char {
    char::from_u32(0x1000 + index as u32).expect("tensor label range")
}

fn label_index(value: char) -> usize {
    value as usize - 0x1000
}

fn single_spin_entropy(state: &[f64], spin: usize) -> f64 {
    let mask = 1usize << spin;
    let mut p0 = 0.0;
    let mut p1 = 0.0;
    let mut off_diagonal = 0.0;
    for basis in 0..state.len() {
        if basis & mask == 0 {
            let paired = basis | mask;
            p0 += state[basis] * state[basis];
            p1 += state[paired] * state[paired];
            off_diagonal += state[basis] * state[paired];
        }
    }
    let radius = ((p0 - p1).powi(2) + 4.0 * off_diagonal * off_diagonal).sqrt();
    [0.5 * (1.0 + radius), 0.5 * (1.0 - radius)]
        .into_iter()
        .filter(|value| *value > 1.0e-14)
        .map(|value| -value * value.ln())
        .sum()
}

fn heisenberg_energy(state: &[f64], bonds: &[QuantumSpinBond], exchange: f64) -> f64 {
    let norm_squared = state
        .iter()
        .map(|amplitude| amplitude * amplitude)
        .sum::<f64>();
    let mut energy = 0.0;
    for bond in bonds {
        let coupling = exchange * bond.multiplicity as f64;
        let mask_a = 1usize << bond.a;
        let mask_b = 1usize << bond.b;
        for basis in 0..state.len() {
            let spin_a = if basis & mask_a == 0 { 1.0 } else { -1.0 };
            let spin_b = if basis & mask_b == 0 { 1.0 } else { -1.0 };
            energy += coupling * spin_a * spin_b * state[basis] * state[basis] / 4.0;
            if spin_a != spin_b {
                energy += coupling * state[basis] * state[basis ^ mask_a ^ mask_b] / 2.0;
            }
        }
    }
    energy / norm_squared.max(1.0e-14)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_spin_thermodynamic_engine::periodic_pyrochlore_model;
    use crate::symmetry_thermodynamic_substrate::SymmetryThermodynamicConfig;

    #[test]
    fn graph_tensor_network_represents_all_periodic_bonds() {
        let (_, bonds) =
            periodic_pyrochlore_model(2, 1, 1, SymmetryThermodynamicConfig::default()).unwrap();
        let network = PyrochloreGraphTensorNetwork::ghz(8, &bonds).unwrap();
        let report = network.report(1.0).unwrap();
        assert_eq!(report.sites, 8);
        assert_eq!(report.tensor_count, 8);
        assert_eq!(report.represented_bonds, 24);
        assert_eq!(report.contraction_steps, 7);
        assert_eq!(report.state_dimension, 256);
        assert_eq!(report.nonzero_amplitudes, 2);
        assert!((report.norm - 1.0).abs() < 1.0e-12);
        assert!((report.single_spin_entropy - std::f64::consts::LN_2).abs() < 1.0e-12);
        assert!((report.heisenberg_energy - 6.0).abs() < 1.0e-12);
        assert!(!report.optimization_available);
    }
}
