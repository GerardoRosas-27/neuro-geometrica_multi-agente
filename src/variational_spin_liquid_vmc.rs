//! Variational Monte Carlo para espines XXZ con ansatz Jastrow complejo.
//!
//! Sigue la estructura usada por NetKet: muestreo de `|ψ|²`, energía local y
//! gradiente covariante. No almacena `2^N` amplitudes y puede superar 20 espines,
//! aunque el ansatz Jastrow es mucho menos expresivo que RBM/PEPS.

use crate::quantum_spin_thermodynamic_engine::QuantumSpinBond;
use crate::symmetry_thermodynamic_substrate::SymmetryThermodynamicSubstrate;
use num_complex::Complex64;

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug)]
pub struct VmcSpinConfig {
    pub exchange: f64,
    pub anisotropy: f64,
    pub symmetry_tying: bool,
    pub all_pair_correlations: bool,
    pub initial_parameter_scale: f64,
    pub parameter_clamp: f64,
    pub sr_diagonal_shift: f64,
    pub seed: u64,
}

impl Default for VmcSpinConfig {
    fn default() -> Self {
        Self {
            exchange: 1.0,
            anisotropy: 1.0,
            symmetry_tying: false,
            all_pair_correlations: true,
            initial_parameter_scale: 0.02,
            parameter_clamp: 2.0,
            sr_diagonal_shift: 0.05,
            seed: 0x5A17_1A5A_2026_0718,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VmcReport {
    pub iteration: usize,
    pub samples: usize,
    pub energy: f64,
    pub energy_imaginary: f64,
    pub variance: f64,
    pub standard_error: f64,
    pub acceptance_rate: f64,
    pub parameter_norm: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VmcOptimizationReport {
    pub initial: VmcReport,
    pub final_report: VmcReport,
    pub best_energy: f64,
    pub accepted_steps: usize,
}

#[derive(Clone, Debug)]
pub struct ComplexJastrowVmc {
    pub geometry: SymmetryThermodynamicSubstrate,
    pub config: VmcSpinConfig,
    pub parameters: Vec<Complex64>,
    pub hamiltonian_bonds: Vec<QuantumSpinBond>,
    ansatz_pairs: Vec<(usize, usize)>,
    parameter_group: Vec<usize>,
    spins: Vec<i8>,
    rng: SplitMix64,
    iteration: usize,
}

impl ComplexJastrowVmc {
    pub fn new(geometry: SymmetryThermodynamicSubstrate, config: VmcSpinConfig) -> Self {
        let bonds = geometry
            .edges
            .iter()
            .map(|edge| QuantumSpinBond {
                a: edge.a,
                b: edge.b,
                multiplicity: 1,
            })
            .collect();
        Self::new_with_bonds(geometry, bonds, config)
    }

    pub fn new_with_bonds(
        geometry: SymmetryThermodynamicSubstrate,
        hamiltonian_bonds: Vec<QuantumSpinBond>,
        config: VmcSpinConfig,
    ) -> Self {
        assert!(
            geometry.vertices.len() % 2 == 0,
            "VMC fixed-magnetization requires an even number of spins"
        );
        let ansatz_pairs = if config.all_pair_correlations {
            let mut pairs = Vec::new();
            for i in 0..geometry.vertices.len() {
                for j in (i + 1)..geometry.vertices.len() {
                    pairs.push((i, j));
                }
            }
            pairs
        } else {
            geometry
                .edges
                .iter()
                .map(|edge| (edge.a.min(edge.b), edge.a.max(edge.b)))
                .collect()
        };
        let parameter_group = if config.symmetry_tying {
            vec![0; ansatz_pairs.len()]
        } else {
            (0..ansatz_pairs.len()).collect()
        };
        let parameter_count = parameter_group.iter().copied().max().unwrap_or(0) + 1;
        let mut rng = SplitMix64::new(config.seed);
        let parameters = (0..parameter_count)
            .map(|_| {
                let real = config.initial_parameter_scale * rng.signed_unit();
                let imaginary = config.initial_parameter_scale * rng.signed_unit();
                Complex64::new(real, imaginary)
            })
            .collect();
        let spins = (0..geometry.vertices.len())
            .map(|spin| if spin % 2 == 0 { 1 } else { -1 })
            .collect();
        Self {
            geometry,
            config,
            parameters,
            hamiltonian_bonds,
            ansatz_pairs,
            parameter_group,
            spins,
            rng,
            iteration: 0,
        }
    }

    pub fn spin_count(&self) -> usize {
        self.spins.len()
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    pub fn configuration(&self) -> &[i8] {
        &self.spins
    }

    pub fn log_wavefunction(&self, spins: &[i8]) -> Complex64 {
        self.ansatz_pairs
            .iter()
            .enumerate()
            .map(|(pair_index, &(a, b))| {
                self.parameters[self.parameter_group[pair_index]]
                    * (spins[a] as f64 * spins[b] as f64)
            })
            .sum()
    }

    pub fn local_energy(&self, spins: &[i8]) -> Complex64 {
        let mut energy = Complex64::new(0.0, 0.0);
        for bond in &self.hamiltonian_bonds {
            let coupling = self.config.exchange * bond.multiplicity as f64;
            let product = spins[bond.a] as f64 * spins[bond.b] as f64;
            energy += coupling * self.config.anisotropy * product / 4.0;
            if spins[bond.a] != spins[bond.b] {
                let ratio = self.log_ratio_exchange(spins, bond.a, bond.b).exp();
                energy += coupling * ratio / 2.0;
            }
        }
        energy
    }

    pub fn exact_variational_energy(&self) -> Option<f64> {
        let spins = self.spin_count();
        if spins > 20 {
            return None;
        }
        let target_up = spins / 2;
        let mut configurations = Vec::new();
        let mut max_log_probability = f64::NEG_INFINITY;
        for basis in 0..(1usize << spins) {
            if basis.count_ones() as usize != target_up {
                continue;
            }
            let configuration = basis_to_spins(basis, spins);
            let log_probability = 2.0 * self.log_wavefunction(&configuration).re;
            max_log_probability = max_log_probability.max(log_probability);
            configurations.push((configuration, log_probability));
        }
        let mut partition = 0.0;
        let mut energy = Complex64::new(0.0, 0.0);
        for (configuration, log_probability) in configurations {
            let weight = (log_probability - max_log_probability).exp();
            partition += weight;
            energy += weight * self.local_energy(&configuration);
        }
        Some((energy / partition.max(EPSILON)).re)
    }

    pub fn sample_report(
        &mut self,
        samples: usize,
        burn_in_sweeps: usize,
        sweeps_between_samples: usize,
    ) -> VmcReport {
        self.sample_and_gradient(samples, burn_in_sweeps, sweeps_between_samples, false, 0.0)
    }

    pub fn optimization_step(
        &mut self,
        samples: usize,
        burn_in_sweeps: usize,
        sweeps_between_samples: usize,
        learning_rate: f64,
    ) -> VmcReport {
        self.sample_and_gradient(
            samples,
            burn_in_sweeps,
            sweeps_between_samples,
            true,
            learning_rate,
        )
    }

    pub fn optimize(
        &mut self,
        iterations: usize,
        samples: usize,
        burn_in_sweeps: usize,
        sweeps_between_samples: usize,
        learning_rate: f64,
    ) -> VmcOptimizationReport {
        let initial = self.sample_report(samples, burn_in_sweeps, sweeps_between_samples);
        let mut best_energy = initial.energy;
        let mut accepted_steps = 0;
        let mut final_report = initial;
        for _ in 0..iterations {
            final_report = self.optimization_step(
                samples,
                burn_in_sweeps,
                sweeps_between_samples,
                learning_rate,
            );
            if final_report.energy < best_energy {
                best_energy = final_report.energy;
                accepted_steps += 1;
            }
        }
        VmcOptimizationReport {
            initial,
            final_report,
            best_energy,
            accepted_steps,
        }
    }

    fn sample_and_gradient(
        &mut self,
        samples: usize,
        burn_in_sweeps: usize,
        sweeps_between_samples: usize,
        update: bool,
        learning_rate: f64,
    ) -> VmcReport {
        let samples = samples.max(1);
        let mut proposed = 0usize;
        let mut accepted = 0usize;
        for _ in 0..burn_in_sweeps {
            let outcome = self.metropolis_sweep();
            proposed += outcome.0;
            accepted += outcome.1;
        }
        let parameter_count = self.parameters.len();
        let mut energy_sum = Complex64::new(0.0, 0.0);
        let mut energy_sq_sum = 0.0;
        let mut observable_sum = vec![0.0; parameter_count];
        let mut observable_sq_sum = vec![0.0; parameter_count];
        let mut observable_energy_sum = vec![Complex64::new(0.0, 0.0); parameter_count];

        for _ in 0..samples {
            for _ in 0..sweeps_between_samples.max(1) {
                let outcome = self.metropolis_sweep();
                proposed += outcome.0;
                accepted += outcome.1;
            }
            let local_energy = self.local_energy(&self.spins);
            let observables = self.log_derivatives(&self.spins);
            energy_sum += local_energy;
            energy_sq_sum += local_energy.norm_sqr();
            for parameter in 0..parameter_count {
                let observable = observables[parameter];
                observable_sum[parameter] += observable;
                observable_sq_sum[parameter] += observable * observable;
                observable_energy_sum[parameter] += observable * local_energy;
            }
        }
        let count = samples as f64;
        let mean_energy = energy_sum / count;
        let variance = (energy_sq_sum / count - mean_energy.norm_sqr()).max(0.0);

        if update {
            for parameter in 0..parameter_count {
                let mean_observable = observable_sum[parameter] / count;
                let covariance =
                    observable_energy_sum[parameter] / count - mean_observable * mean_energy;
                let observable_variance =
                    observable_sq_sum[parameter] / count - mean_observable * mean_observable;
                let denominator = observable_variance.max(0.0) + self.config.sr_diagonal_shift;
                let gradient_real = 2.0 * covariance.re / denominator;
                let gradient_imaginary = 2.0 * covariance.im / denominator;
                self.parameters[parameter].re -= learning_rate * gradient_real;
                self.parameters[parameter].im -= learning_rate * gradient_imaginary;
                self.parameters[parameter].re = self.parameters[parameter]
                    .re
                    .clamp(-self.config.parameter_clamp, self.config.parameter_clamp);
                self.parameters[parameter].im = self.parameters[parameter]
                    .im
                    .clamp(-self.config.parameter_clamp, self.config.parameter_clamp);
            }
        }
        self.iteration += usize::from(update);
        VmcReport {
            iteration: self.iteration,
            samples,
            energy: mean_energy.re,
            energy_imaginary: mean_energy.im,
            variance,
            standard_error: (variance / count).sqrt(),
            acceptance_rate: accepted as f64 / proposed.max(1) as f64,
            parameter_norm: self
                .parameters
                .iter()
                .map(Complex64::norm_sqr)
                .sum::<f64>()
                .sqrt(),
        }
    }

    fn metropolis_sweep(&mut self) -> (usize, usize) {
        let proposals = self.spin_count();
        let mut accepted = 0;
        for _ in 0..proposals {
            let i = self.rng.index(self.spin_count());
            let mut j = self.rng.index(self.spin_count());
            for _ in 0..self.spin_count() {
                if self.spins[i] != self.spins[j] {
                    break;
                }
                j = (j + 1) % self.spin_count();
            }
            if self.spins[i] == self.spins[j] {
                continue;
            }
            let ratio = self.log_ratio_exchange(&self.spins, i, j);
            let probability = (2.0 * ratio.re).exp().min(1.0);
            if self.rng.unit() < probability {
                self.spins[i] = -self.spins[i];
                self.spins[j] = -self.spins[j];
                accepted += 1;
            }
        }
        (proposals, accepted)
    }

    fn log_ratio_exchange(&self, spins: &[i8], i: usize, j: usize) -> Complex64 {
        self.ansatz_pairs
            .iter()
            .enumerate()
            .filter_map(|(pair_index, &(a, b))| {
                let touches_i = a == i || b == i;
                let touches_j = a == j || b == j;
                if !touches_i && !touches_j {
                    return None;
                }
                let old = spins[a] as f64 * spins[b] as f64;
                let new_a = if a == i || a == j {
                    -spins[a]
                } else {
                    spins[a]
                };
                let new_b = if b == i || b == j {
                    -spins[b]
                } else {
                    spins[b]
                };
                let new = new_a as f64 * new_b as f64;
                Some(self.parameters[self.parameter_group[pair_index]] * (new - old))
            })
            .sum()
    }

    fn log_derivatives(&self, spins: &[i8]) -> Vec<f64> {
        let mut derivatives = vec![0.0; self.parameters.len()];
        for (pair_index, &(a, b)) in self.ansatz_pairs.iter().enumerate() {
            derivatives[self.parameter_group[pair_index]] += spins[a] as f64 * spins[b] as f64;
        }
        derivatives
    }
}

fn basis_to_spins(basis: usize, spins: usize) -> Vec<i8> {
    (0..spins)
        .map(|spin| if (basis >> spin) & 1 == 0 { 1 } else { -1 })
        .collect()
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn signed_unit(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_spin_thermodynamic_engine::{
        periodic_pyrochlore_model, QuantumSpinConfig, QuantumSpinThermodynamicEngine,
    };
    use crate::symmetry_thermodynamic_substrate::SymmetryThermodynamicConfig;

    fn vmc(spins: usize) -> ComplexJastrowVmc {
        let dimensions = match spins {
            8 => (2, 1, 1),
            12 => (3, 1, 1),
            16 => (2, 2, 1),
            32 => (2, 2, 2),
            _ => panic!("unsupported fixture"),
        };
        let (geometry, bonds) = periodic_pyrochlore_model(
            dimensions.0,
            dimensions.1,
            dimensions.2,
            SymmetryThermodynamicConfig::default(),
        )
        .unwrap();
        ComplexJastrowVmc::new_with_bonds(geometry, bonds, VmcSpinConfig::default())
    }

    #[test]
    fn log_ratio_matches_direct_wavefunction_difference() {
        let vmc = vmc(8);
        let before = vmc.log_wavefunction(vmc.configuration());
        let mut exchanged = vmc.configuration().to_vec();
        let i = 0;
        let j = 1;
        exchanged[i] = -exchanged[i];
        exchanged[j] = -exchanged[j];
        let direct = vmc.log_wavefunction(&exchanged) - before;
        let local = vmc.log_ratio_exchange(vmc.configuration(), i, j);
        assert!((direct - local).norm() < 1.0e-12);
    }

    #[test]
    fn monte_carlo_energy_matches_exact_variational_energy() {
        let mut vmc = vmc(8);
        let exact = vmc.exact_variational_energy().unwrap();
        let sampled = vmc.sample_report(20_000, 100, 2);
        assert!((sampled.energy - exact).abs() < 5.0 * sampled.standard_error + 0.03);
        assert!(sampled.acceptance_rate > 0.0 && sampled.acceptance_rate <= 1.0);
    }

    #[test]
    fn variational_energy_respects_exact_ground_bound() {
        let vmc = vmc(8);
        let exact_variational = vmc.exact_variational_energy().unwrap();
        let exact_engine = QuantumSpinThermodynamicEngine::new_neel_with_bonds(
            vmc.geometry.clone(),
            vmc.hamiltonian_bonds.clone(),
            QuantumSpinConfig {
                spin_lattice_alpha: 0.0,
                ..QuantumSpinConfig::default()
            },
        )
        .unwrap();
        let ground = exact_engine.lanczos_spectrum(36).ground_energy;
        assert!(exact_variational + 1.0e-8 >= ground);
    }

    #[test]
    fn thirty_two_spin_vmc_uses_polynomial_state() {
        let vmc = vmc(32);
        assert_eq!(vmc.spin_count(), 32);
        assert!(vmc.parameter_count() <= 32 * 31 / 2);
        assert_eq!(vmc.configuration().len(), 32);
    }
}
