//! Backend cuántico exacto pequeño para espines 1/2 sobre una malla simplicial.
//!
//! Usa un vector de estado de `2^N` amplitudes complejas y puertas locales XXZ.
//! No construye el Hamiltoniano denso, pero el coste de memoria sigue siendo
//! exponencial. Está pensado para validar 4–16 espines, no para producción.

use crate::symmetry_thermodynamic_substrate::{
    SymmetrySubstrateError, SymmetryThermodynamicConfig, SymmetryThermodynamicSubstrate,
    Tetrahedron, Vec3,
};
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::fmt;

const EPSILON: f64 = 1.0e-14;

#[derive(Clone, Copy, Debug)]
pub struct QuantumSpinConfig {
    pub exchange: f64,
    pub anisotropy: f64,
    pub magnetic_field_z: f64,
    pub spin_lattice_alpha: f64,
    pub equilibrium_length: f64,
    pub real_time_step: f64,
    pub imaginary_time_step: f64,
    pub entanglement_witness_threshold: f64,
    pub max_spins: usize,
}

impl Default for QuantumSpinConfig {
    fn default() -> Self {
        Self {
            exchange: 1.0,
            anisotropy: 1.0,
            magnetic_field_z: 0.0,
            spin_lattice_alpha: 1.5,
            equilibrium_length: 1.0,
            real_time_step: 0.015,
            imaginary_time_step: 0.01,
            entanglement_witness_threshold: 1.0e-5,
            max_spins: 16,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantumSpinBond {
    pub a: usize,
    pub b: usize,
    pub multiplicity: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantumSpinError {
    EmptyGeometry,
    TooManySpins { requested: usize, maximum: usize },
    BasisOutOfRange { basis: usize, dimension: usize },
    StateDimensionMismatch { expected: usize, received: usize },
}

impl fmt::Display for QuantumSpinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGeometry => write!(f, "la geometría no contiene espines"),
            Self::TooManySpins { requested, maximum } => write!(
                f,
                "{requested} espines exceden el máximo exacto configurado de {maximum}"
            ),
            Self::BasisOutOfRange { basis, dimension } => {
                write!(f, "estado base {basis} fuera de dimensión {dimension}")
            }
            Self::StateDimensionMismatch { expected, received } => write!(
                f,
                "estado cuántico de dimensión {received}; se esperaba {expected}"
            ),
        }
    }
}

impl std::error::Error for QuantumSpinError {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QuantumEntanglementLink {
    pub a: usize,
    pub b: usize,
    pub spin_correlation: f64,
    /// Testigo suficiente para Heisenberg isotrópico:
    /// estados separables cumplen `<S_i·S_j> >= -1/4`.
    pub witness: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QuantumSpinReport {
    pub tick: u64,
    pub spins: usize,
    pub hilbert_dimension: usize,
    pub norm: f64,
    pub energy: f64,
    pub energy_per_edge: f64,
    pub magnetization_z: f64,
    pub mean_single_spin_entropy: f64,
    pub half_system_renyi2: f64,
    pub entangled_edges: usize,
    pub max_entanglement_witness: f64,
    pub mean_spin_correlation: f64,
    pub correlation_variance: f64,
    pub geometry_symmetry: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LanczosSpectrum {
    pub ground_energy: f64,
    pub first_excited_energy: f64,
    pub gap: f64,
    pub iterations: usize,
}

#[derive(Clone, Debug)]
pub struct QuantumSpinThermodynamicEngine {
    pub geometry: SymmetryThermodynamicSubstrate,
    pub config: QuantumSpinConfig,
    pub bonds: Vec<QuantumSpinBond>,
    state: Vec<Complex64>,
    tick: u64,
}

impl QuantumSpinThermodynamicEngine {
    pub fn new_neel(
        geometry: SymmetryThermodynamicSubstrate,
        config: QuantumSpinConfig,
    ) -> Result<Self, QuantumSpinError> {
        let bonds = geometry
            .edges
            .iter()
            .map(|edge| QuantumSpinBond {
                a: edge.a,
                b: edge.b,
                multiplicity: 1,
            })
            .collect();
        Self::new_neel_with_bonds(geometry, bonds, config)
    }

    pub fn new_neel_with_bonds(
        geometry: SymmetryThermodynamicSubstrate,
        bonds: Vec<QuantumSpinBond>,
        config: QuantumSpinConfig,
    ) -> Result<Self, QuantumSpinError> {
        let spins = geometry.vertices.len();
        validate_size(spins, config.max_spins)?;
        let basis = (0..spins)
            .filter(|spin| spin % 2 == 1)
            .fold(0usize, |bits, spin| bits | (1usize << spin));
        Self::new_basis_with_bonds(geometry, bonds, config, basis)
    }

    pub fn new_basis(
        geometry: SymmetryThermodynamicSubstrate,
        config: QuantumSpinConfig,
        basis: usize,
    ) -> Result<Self, QuantumSpinError> {
        let bonds = geometry
            .edges
            .iter()
            .map(|edge| QuantumSpinBond {
                a: edge.a,
                b: edge.b,
                multiplicity: 1,
            })
            .collect();
        Self::new_basis_with_bonds(geometry, bonds, config, basis)
    }

    pub fn new_basis_with_bonds(
        geometry: SymmetryThermodynamicSubstrate,
        bonds: Vec<QuantumSpinBond>,
        config: QuantumSpinConfig,
        basis: usize,
    ) -> Result<Self, QuantumSpinError> {
        let spins = geometry.vertices.len();
        validate_size(spins, config.max_spins)?;
        let dimension = 1usize << spins;
        if basis >= dimension {
            return Err(QuantumSpinError::BasisOutOfRange { basis, dimension });
        }
        let mut state = vec![Complex64::new(0.0, 0.0); dimension];
        state[basis] = Complex64::new(1.0, 0.0);
        Ok(Self {
            geometry,
            config: sanitize_config(config),
            bonds,
            state,
            tick: 0,
        })
    }

    pub fn amplitudes(&self) -> &[Complex64] {
        &self.state
    }

    pub fn set_amplitudes(&mut self, amplitudes: &[Complex64]) -> Result<(), QuantumSpinError> {
        if amplitudes.len() != self.state.len() {
            return Err(QuantumSpinError::StateDimensionMismatch {
                expected: self.state.len(),
                received: amplitudes.len(),
            });
        }
        self.state.copy_from_slice(amplitudes);
        normalize(&mut self.state);
        Ok(())
    }

    pub fn spin_count(&self) -> usize {
        self.geometry.vertices.len()
    }

    pub fn hilbert_dimension(&self) -> usize {
        self.state.len()
    }

    pub fn apply_local_phase_pulse(&mut self, spin: usize, angle: f64) -> bool {
        if spin >= self.spin_count() || !angle.is_finite() {
            return false;
        }
        apply_z_field_unitary(&mut self.state, spin, 1.0, angle);
        true
    }

    pub fn apply_exchange_pulse(&mut self, bond: usize, duration: f64) -> bool {
        if bond >= self.bonds.len() || !duration.is_finite() {
            return false;
        }
        let (a, b, coupling) = self.bond_coupling(bond);
        apply_xxz_unitary(
            &mut self.state,
            a,
            b,
            coupling,
            self.config.anisotropy,
            duration,
        );
        true
    }

    pub fn real_time_step(&mut self) -> QuantumSpinReport {
        let dt = self.config.real_time_step;
        for bond in 0..self.bonds.len() {
            let (a, b, coupling) = self.bond_coupling(bond);
            apply_xxz_unitary(&mut self.state, a, b, coupling, self.config.anisotropy, dt);
        }
        if self.config.magnetic_field_z.abs() > EPSILON {
            for spin in 0..self.spin_count() {
                apply_z_field_unitary(&mut self.state, spin, self.config.magnetic_field_z, dt);
            }
        }
        self.tick += 1;
        self.report()
    }

    /// Evolución euclidiana normalizada hacia estados de menor energía.
    pub fn imaginary_time_step(&mut self) -> QuantumSpinReport {
        let tau = self.config.imaginary_time_step;
        for bond in 0..self.bonds.len() {
            let (a, b, coupling) = self.bond_coupling(bond);
            apply_xxz_imaginary(&mut self.state, a, b, coupling, self.config.anisotropy, tau);
        }
        if self.config.magnetic_field_z.abs() > EPSILON {
            for spin in 0..self.spin_count() {
                apply_z_field_imaginary(&mut self.state, spin, self.config.magnetic_field_z, tau);
            }
        }
        normalize(&mut self.state);
        self.tick += 1;
        self.report()
    }

    pub fn cool(&mut self, steps: usize) -> QuantumSpinReport {
        let mut report = self.report();
        for _ in 0..steps {
            report = self.imaginary_time_step();
        }
        report
    }

    pub fn quantum_entanglement_links(&self) -> Vec<QuantumEntanglementLink> {
        if (self.config.anisotropy - 1.0).abs() > 1.0e-9 {
            return Vec::new();
        }
        self.bonds
            .iter()
            .filter_map(|bond| {
                let correlation = spin_dot_correlation(&self.state, bond.a, bond.b);
                let witness = (-0.25 - correlation).max(0.0);
                (witness >= self.config.entanglement_witness_threshold).then_some(
                    QuantumEntanglementLink {
                        a: bond.a,
                        b: bond.b,
                        spin_correlation: correlation,
                        witness,
                    },
                )
            })
            .collect()
    }

    /// Retroacción de Born–Oppenheimer simplificada usando
    /// `F_e = α J_e <S_i·S_j>`. Conserva el centroide por pares.
    pub fn spin_lattice_backreaction(&mut self, rate: f64) -> f64 {
        let mut displacements = vec![Vec3::default(); self.spin_count()];
        let mut max_force: f64 = 0.0;
        for (bond_index, bond) in self.bonds.iter().enumerate() {
            let (_, _, coupling) = self.bond_coupling(bond_index);
            let correlation = spin_dot_correlation(&self.state, bond.a, bond.b);
            let force = self.config.spin_lattice_alpha * coupling * correlation;
            let delta =
                self.geometry.vertices[bond.b].position - self.geometry.vertices[bond.a].position;
            let direction = delta / delta.norm().max(EPSILON);
            let displacement = direction * (rate * force);
            displacements[bond.a] = displacements[bond.a] + displacement;
            displacements[bond.b] = displacements[bond.b] - displacement;
            max_force = max_force.max(force.abs());
        }
        for (vertex, displacement) in self.geometry.vertices.iter_mut().zip(displacements) {
            vertex.position = vertex.position + displacement;
        }
        max_force
    }

    /// Aplica H sin construir una matriz `2^N × 2^N`.
    pub fn hamiltonian_action(&self, input: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(input.len(), self.state.len());
        let mut output = vec![Complex64::new(0.0, 0.0); input.len()];
        for bond_index in 0..self.bonds.len() {
            let (i, j, coupling) = self.bond_coupling(bond_index);
            let bit_i = 1usize << i;
            let bit_j = 1usize << j;
            for basis in 0..input.len() {
                let zi = if basis & bit_i == 0 { 0.5 } else { -0.5 };
                let zj = if basis & bit_j == 0 { 0.5 } else { -0.5 };
                output[basis] += input[basis] * (coupling * self.config.anisotropy * zi * zj);
                if (basis & bit_i == 0) != (basis & bit_j == 0) {
                    output[basis ^ bit_i ^ bit_j] += input[basis] * (coupling / 2.0);
                }
            }
        }
        if self.config.magnetic_field_z.abs() > EPSILON {
            for basis in 0..input.len() {
                let magnetization = (0..self.spin_count())
                    .map(|spin| if bit(basis, spin) == 0 { 0.5 } else { -0.5 })
                    .sum::<f64>();
                output[basis] += input[basis] * (self.config.magnetic_field_z * magnetization);
            }
        }
        output
    }

    /// Espectro extremo aproximado por Lanczos con reortogonalización completa.
    pub fn lanczos_spectrum(&self, iterations: usize) -> LanczosSpectrum {
        let iterations = iterations.clamp(2, self.state.len());
        let mut q = deterministic_probe(self.state.len());
        normalize(&mut q);
        let mut q_previous = vec![Complex64::new(0.0, 0.0); self.state.len()];
        let mut basis_vectors = Vec::<Vec<Complex64>>::with_capacity(iterations);
        let mut alphas = Vec::with_capacity(iterations);
        let mut betas = Vec::with_capacity(iterations.saturating_sub(1));
        let mut beta_previous = 0.0;

        for step in 0..iterations {
            basis_vectors.push(q.clone());
            let mut residual = self.hamiltonian_action(&q);
            let alpha = inner_product(&q, &residual).re;
            for index in 0..residual.len() {
                residual[index] -= alpha * q[index] + beta_previous * q_previous[index];
            }
            for vector in &basis_vectors {
                let projection = inner_product(vector, &residual);
                for index in 0..residual.len() {
                    residual[index] -= projection * vector[index];
                }
            }
            let beta = state_norm(&residual);
            alphas.push(alpha);
            if step + 1 >= iterations || beta <= 1.0e-11 {
                break;
            }
            betas.push(beta);
            q_previous = q;
            q = residual;
            for amplitude in &mut q {
                *amplitude /= beta;
            }
            beta_previous = beta;
        }

        let eigenvalues = symmetric_tridiagonal_eigenvalues(&alphas, &betas);
        let ground = eigenvalues.first().copied().unwrap_or(0.0);
        let first = eigenvalues.get(1).copied().unwrap_or(ground);
        LanczosSpectrum {
            ground_energy: ground,
            first_excited_energy: first,
            gap: (first - ground).max(0.0),
            iterations: alphas.len(),
        }
    }

    pub fn static_structure_factor(&self, wave_vector: Vec3) -> f64 {
        let spins = self.spin_count();
        let mut total = 0.0;
        for i in 0..spins {
            for j in 0..spins {
                let correlation = if i == j {
                    0.75
                } else {
                    spin_dot_correlation(&self.state, i, j)
                };
                let displacement =
                    self.geometry.vertices[i].position - self.geometry.vertices[j].position;
                total += (wave_vector.dot(displacement)).cos() * correlation;
            }
        }
        total / spins.max(1) as f64
    }

    pub fn maximum_structure_factor(&self, wave_vectors: &[Vec3]) -> f64 {
        wave_vectors
            .iter()
            .map(|wave_vector| self.static_structure_factor(*wave_vector))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn report(&self) -> QuantumSpinReport {
        let norm = state_norm(&self.state);
        let mut energy = 0.0;
        let physical_bonds = self
            .bonds
            .iter()
            .map(|bond| bond.multiplicity as usize)
            .sum::<usize>();
        let mut correlations = Vec::with_capacity(physical_bonds);
        for (bond_index, bond) in self.bonds.iter().enumerate() {
            let (_, _, coupling) = self.bond_coupling(bond_index);
            let xy = spin_xy_correlation(&self.state, bond.a, bond.b);
            let zz = spin_zz_correlation(&self.state, bond.a, bond.b);
            energy += coupling * (xy + self.config.anisotropy * zz);
            correlations.extend(std::iter::repeat_n(xy + zz, bond.multiplicity as usize));
        }
        let magnetization_z = (0..self.spin_count())
            .map(|spin| spin_z_expectation(&self.state, spin))
            .sum::<f64>()
            / self.spin_count().max(1) as f64;
        energy += self.config.magnetic_field_z
            * (0..self.spin_count())
                .map(|spin| spin_z_expectation(&self.state, spin))
                .sum::<f64>();
        let mean_single_spin_entropy = (0..self.spin_count())
            .map(|spin| single_spin_entropy(&self.state, spin))
            .sum::<f64>()
            / self.spin_count().max(1) as f64;
        let mean_correlation = correlations.iter().sum::<f64>() / correlations.len().max(1) as f64;
        let correlation_variance = correlations
            .iter()
            .map(|correlation| (correlation - mean_correlation).powi(2))
            .sum::<f64>()
            / correlations.len().max(1) as f64;
        let links = self.quantum_entanglement_links();
        QuantumSpinReport {
            tick: self.tick,
            spins: self.spin_count(),
            hilbert_dimension: self.hilbert_dimension(),
            norm,
            energy,
            energy_per_edge: energy / physical_bonds.max(1) as f64,
            magnetization_z,
            mean_single_spin_entropy,
            half_system_renyi2: subsystem_renyi2(&self.state, self.spin_count() / 2),
            entangled_edges: links.len(),
            max_entanglement_witness: links.iter().map(|link| link.witness).fold(0.0, f64::max),
            mean_spin_correlation: mean_correlation,
            correlation_variance,
            geometry_symmetry: self.geometry.metrics().symmetry_score,
        }
    }

    fn bond_coupling(&self, bond_index: usize) -> (usize, usize, f64) {
        let bond = self.bonds[bond_index];
        let length = (self.geometry.vertices[bond.a].position
            - self.geometry.vertices[bond.b].position)
            .norm();
        let coupling = self.config.exchange
            * bond.multiplicity as f64
            * (-self.config.spin_lattice_alpha * (length - self.config.equilibrium_length)).exp();
        (bond.a, bond.b, coupling)
    }
}

/// Cluster periódico pyrochlore como line graph de una red diamante.
/// Cada celda primitiva aporta cuatro sitios de espín y dos tetraedros.
pub fn periodic_pyrochlore_geometry(
    nx: usize,
    ny: usize,
    nz: usize,
    config: SymmetryThermodynamicConfig,
) -> Result<SymmetryThermodynamicSubstrate, SymmetrySubstrateError> {
    let cells = nx.saturating_mul(ny).saturating_mul(nz);
    if cells == 0 {
        return SymmetryThermodynamicSubstrate::new(Vec::new(), Vec::new(), 1.0, config);
    }
    let primitive = [
        Vec3::new(0.0, 0.5, 0.5),
        Vec3::new(0.5, 0.0, 0.5),
        Vec3::new(0.5, 0.5, 0.0),
    ];
    let diamond_directions = [
        Vec3::new(0.25, 0.25, 0.25),
        Vec3::new(0.25, -0.25, -0.25),
        Vec3::new(-0.25, 0.25, -0.25),
        Vec3::new(-0.25, -0.25, 0.25),
    ];
    let mut positions = Vec::with_capacity(cells * 4);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let origin =
                    primitive[0] * x as f64 + primitive[1] * y as f64 + primitive[2] * z as f64;
                for direction in diamond_directions {
                    positions.push(origin + direction / 2.0);
                }
            }
        }
    }

    let site = |x: usize, y: usize, z: usize, direction: usize| {
        (((z % nz) * ny + (y % ny)) * nx + (x % nx)) * 4 + direction
    };
    let mut tetrahedra = Vec::with_capacity(cells * 2);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                tetrahedra.push(Tetrahedron([
                    site(x, y, z, 0),
                    site(x, y, z, 1),
                    site(x, y, z, 2),
                    site(x, y, z, 3),
                ]));
                tetrahedra.push(Tetrahedron([
                    site(x, y, z, 0),
                    site((x + 1) % nx, y, z, 1),
                    site(x, (y + 1) % ny, z, 2),
                    site(x, y, (z + 1) % nz, 3),
                ]));
            }
        }
    }
    let target_length = (0.125_f64).sqrt();
    SymmetryThermodynamicSubstrate::new(positions, tetrahedra, target_length, config)
}

pub fn periodic_pyrochlore_model(
    nx: usize,
    ny: usize,
    nz: usize,
    config: SymmetryThermodynamicConfig,
) -> Result<(SymmetryThermodynamicSubstrate, Vec<QuantumSpinBond>), SymmetrySubstrateError> {
    let geometry = periodic_pyrochlore_geometry(nx, ny, nz, config)?;
    let mut multiplicities = BTreeMap::<(usize, usize), u32>::new();
    for tetrahedron in &geometry.tetrahedra {
        for i in 0..4 {
            for j in (i + 1)..4 {
                let a = tetrahedron.0[i].min(tetrahedron.0[j]);
                let b = tetrahedron.0[i].max(tetrahedron.0[j]);
                *multiplicities.entry((a, b)).or_default() += 1;
            }
        }
    }
    let bonds = multiplicities
        .into_iter()
        .map(|((a, b), multiplicity)| QuantumSpinBond { a, b, multiplicity })
        .collect();
    Ok((geometry, bonds))
}

fn sanitize_config(config: QuantumSpinConfig) -> QuantumSpinConfig {
    QuantumSpinConfig {
        exchange: config.exchange,
        anisotropy: config.anisotropy,
        spin_lattice_alpha: config.spin_lattice_alpha.max(0.0),
        equilibrium_length: config.equilibrium_length.abs().max(EPSILON),
        real_time_step: config.real_time_step.abs().max(EPSILON),
        imaginary_time_step: config.imaginary_time_step.abs().max(EPSILON),
        entanglement_witness_threshold: config.entanglement_witness_threshold.max(0.0),
        max_spins: config.max_spins.clamp(1, 20),
        ..config
    }
}

fn validate_size(spins: usize, maximum: usize) -> Result<(), QuantumSpinError> {
    if spins == 0 {
        return Err(QuantumSpinError::EmptyGeometry);
    }
    if spins > maximum {
        return Err(QuantumSpinError::TooManySpins {
            requested: spins,
            maximum,
        });
    }
    Ok(())
}

fn apply_xxz_unitary(
    state: &mut [Complex64],
    i: usize,
    j: usize,
    coupling: f64,
    anisotropy: f64,
    dt: f64,
) {
    let diagonal_parallel = coupling * anisotropy / 4.0;
    let diagonal_anti = -diagonal_parallel;
    let exchange = coupling / 2.0;
    let parallel_phase = Complex64::from_polar(1.0, -dt * diagonal_parallel);
    let anti_phase = Complex64::from_polar(1.0, -dt * diagonal_anti);
    let cosine = (dt * exchange).cos();
    let minus_i_sine = Complex64::new(0.0, -(dt * exchange).sin());
    for_each_pair(state.len(), i, j, |i00, i01, i10, i11| {
        let a01 = state[i01];
        let a10 = state[i10];
        state[i00] *= parallel_phase;
        state[i11] *= parallel_phase;
        state[i01] = anti_phase * (cosine * a01 + minus_i_sine * a10);
        state[i10] = anti_phase * (minus_i_sine * a01 + cosine * a10);
    });
}

fn apply_xxz_imaginary(
    state: &mut [Complex64],
    i: usize,
    j: usize,
    coupling: f64,
    anisotropy: f64,
    tau: f64,
) {
    let diagonal_parallel = coupling * anisotropy / 4.0;
    let diagonal_anti = -diagonal_parallel;
    let exchange = coupling / 2.0;
    let parallel = (-tau * diagonal_parallel).exp();
    let anti = (-tau * diagonal_anti).exp();
    let cosine = (tau * exchange).cosh();
    let minus_sine = -(tau * exchange).sinh();
    for_each_pair(state.len(), i, j, |i00, i01, i10, i11| {
        let a01 = state[i01];
        let a10 = state[i10];
        state[i00] *= parallel;
        state[i11] *= parallel;
        state[i01] = anti * (cosine * a01 + minus_sine * a10);
        state[i10] = anti * (minus_sine * a01 + cosine * a10);
    });
}

fn apply_z_field_unitary(state: &mut [Complex64], spin: usize, field: f64, dt: f64) {
    let up = Complex64::from_polar(1.0, -dt * field / 2.0);
    let down = Complex64::from_polar(1.0, dt * field / 2.0);
    for (basis, amplitude) in state.iter_mut().enumerate() {
        *amplitude *= if bit(basis, spin) == 0 { up } else { down };
    }
}

fn apply_z_field_imaginary(state: &mut [Complex64], spin: usize, field: f64, tau: f64) {
    let up = (-tau * field / 2.0).exp();
    let down = (tau * field / 2.0).exp();
    for (basis, amplitude) in state.iter_mut().enumerate() {
        *amplitude *= if bit(basis, spin) == 0 { up } else { down };
    }
}

fn for_each_pair(
    dimension: usize,
    i: usize,
    j: usize,
    mut operation: impl FnMut(usize, usize, usize, usize),
) {
    let bit_i = 1usize << i;
    let bit_j = 1usize << j;
    for basis in 0..dimension {
        if basis & bit_i == 0 && basis & bit_j == 0 {
            operation(basis, basis | bit_j, basis | bit_i, basis | bit_i | bit_j);
        }
    }
}

fn state_norm(state: &[Complex64]) -> f64 {
    state.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

fn normalize(state: &mut [Complex64]) {
    let norm = state_norm(state).max(EPSILON);
    for amplitude in state {
        *amplitude /= norm;
    }
}

fn spin_z_expectation(state: &[Complex64], spin: usize) -> f64 {
    state
        .iter()
        .enumerate()
        .map(|(basis, amplitude)| {
            let z = if bit(basis, spin) == 0 { 0.5 } else { -0.5 };
            z * amplitude.norm_sqr()
        })
        .sum()
}

fn spin_zz_correlation(state: &[Complex64], i: usize, j: usize) -> f64 {
    state
        .iter()
        .enumerate()
        .map(|(basis, amplitude)| {
            let zi = if bit(basis, i) == 0 { 0.5 } else { -0.5 };
            let zj = if bit(basis, j) == 0 { 0.5 } else { -0.5 };
            zi * zj * amplitude.norm_sqr()
        })
        .sum()
}

fn spin_xy_correlation(state: &[Complex64], i: usize, j: usize) -> f64 {
    let mut correlation = 0.0;
    for_each_pair(state.len(), i, j, |_, i01, i10, _| {
        correlation += (state[i01].conj() * state[i10]).re;
    });
    correlation
}

fn spin_dot_correlation(state: &[Complex64], i: usize, j: usize) -> f64 {
    spin_xy_correlation(state, i, j) + spin_zz_correlation(state, i, j)
}

fn single_spin_entropy(state: &[Complex64], spin: usize) -> f64 {
    let bit_mask = 1usize << spin;
    let mut p0 = 0.0;
    let mut p1 = 0.0;
    let mut off_diagonal = Complex64::new(0.0, 0.0);
    for basis in 0..state.len() {
        if basis & bit_mask == 0 {
            let paired = basis | bit_mask;
            p0 += state[basis].norm_sqr();
            p1 += state[paired].norm_sqr();
            off_diagonal += state[basis] * state[paired].conj();
        }
    }
    let radius = ((p0 - p1).powi(2) + 4.0 * off_diagonal.norm_sqr())
        .max(0.0)
        .sqrt();
    let eigenvalues = [
        ((1.0 + radius) / 2.0).clamp(0.0, 1.0),
        ((1.0 - radius) / 2.0).clamp(0.0, 1.0),
    ];
    eigenvalues
        .into_iter()
        .filter(|value| *value > EPSILON)
        .map(|value| -value * value.ln())
        .sum()
}

fn subsystem_renyi2(state: &[Complex64], subsystem_spins: usize) -> f64 {
    if subsystem_spins == 0 {
        return 0.0;
    }
    let dimension_a = 1usize << subsystem_spins;
    let dimension_b = state.len() / dimension_a;
    let mut purity = 0.0;
    for a in 0..dimension_a {
        for a_prime in 0..dimension_a {
            let mut rho = Complex64::new(0.0, 0.0);
            for b in 0..dimension_b {
                rho += state[a | (b << subsystem_spins)]
                    * state[a_prime | (b << subsystem_spins)].conj();
            }
            purity += rho.norm_sqr();
        }
    }
    -purity.max(EPSILON).ln()
}

fn bit(value: usize, position: usize) -> usize {
    (value >> position) & 1
}

fn deterministic_probe(dimension: usize) -> Vec<Complex64> {
    let mut state = 0x51A7_E5E1_DA7A_2026_u64;
    (0..dimension)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            value ^= value >> 31;
            let real = ((value & 0xffff_ffff) as f64 / u32::MAX as f64) - 0.5;
            let imaginary = ((value >> 32) as f64 / u32::MAX as f64) - 0.5;
            Complex64::new(real, imaginary)
        })
        .collect()
}

fn inner_product(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * right)
        .sum()
}

fn symmetric_tridiagonal_eigenvalues(diagonal: &[f64], off_diagonal: &[f64]) -> Vec<f64> {
    let size = diagonal.len();
    let mut matrix = vec![vec![0.0; size]; size];
    for index in 0..size {
        matrix[index][index] = diagonal[index];
        if index + 1 < size {
            let value = off_diagonal.get(index).copied().unwrap_or(0.0);
            matrix[index][index + 1] = value;
            matrix[index + 1][index] = value;
        }
    }
    for _ in 0..(size * size * 50).max(1) {
        let mut p = 0;
        let mut q = 0;
        let mut maximum = 0.0;
        for row in 0..size {
            for column in (row + 1)..size {
                if matrix[row][column].abs() > maximum {
                    maximum = matrix[row][column].abs();
                    p = row;
                    q = column;
                }
            }
        }
        if maximum < 1.0e-12 {
            break;
        }
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let cosine = angle.cos();
        let sine = angle.sin();
        let app = cosine * cosine * matrix[p][p] - 2.0 * sine * cosine * matrix[p][q]
            + sine * sine * matrix[q][q];
        let aqq = sine * sine * matrix[p][p]
            + 2.0 * sine * cosine * matrix[p][q]
            + cosine * cosine * matrix[q][q];
        for index in 0..size {
            if index != p && index != q {
                let aip = matrix[index][p];
                let aiq = matrix[index][q];
                matrix[index][p] = cosine * aip - sine * aiq;
                matrix[p][index] = matrix[index][p];
                matrix[index][q] = sine * aip + cosine * aiq;
                matrix[q][index] = matrix[index][q];
            }
        }
        matrix[p][p] = app;
        matrix[q][q] = aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
    }
    let mut eigenvalues = (0..size)
        .map(|index| matrix[index][index])
        .collect::<Vec<_>>();
    eigenvalues.sort_by(f64::total_cmp);
    eigenvalues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetry_thermodynamic_substrate::SymmetryThermodynamicConfig;

    fn engine() -> QuantumSpinThermodynamicEngine {
        let geometry = SymmetryThermodynamicSubstrate::regular_tetrahedron(
            1.0,
            SymmetryThermodynamicConfig::default(),
        )
        .unwrap();
        QuantumSpinThermodynamicEngine::new_neel(geometry, QuantumSpinConfig::default()).unwrap()
    }

    #[test]
    fn product_state_has_unit_norm_and_zero_entanglement_entropy() {
        let engine = engine();
        let report = engine.report();
        assert!((report.norm - 1.0).abs() < 1.0e-12);
        assert!(report.mean_single_spin_entropy < 1.0e-12);
        assert!(report.half_system_renyi2 < 1.0e-12);
    }

    #[test]
    fn local_unitaries_preserve_norm() {
        let mut engine = engine();
        for _ in 0..200 {
            engine.real_time_step();
        }
        assert!((engine.report().norm - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn imaginary_time_lowers_energy_and_generates_entanglement() {
        let mut engine = engine();
        let before = engine.report();
        let after = engine.cool(500);
        assert!(after.energy < before.energy);
        assert!(after.mean_single_spin_entropy > 0.1, "{after:?}");
        assert!(after.half_system_renyi2 > 0.05, "{after:?}");
        assert!(after.entangled_edges > 0, "{after:?}");
    }

    #[test]
    fn spin_lattice_backreaction_preserves_centroid() {
        let mut engine = engine();
        engine.cool(100);
        let before = centroid(&engine.geometry);
        let force = engine.spin_lattice_backreaction(1.0e-4);
        let after = centroid(&engine.geometry);
        assert!(force.is_finite() && force > 0.0);
        assert!((after - before).norm() < 1.0e-12);
    }

    #[test]
    fn matrix_free_hamiltonian_and_lanczos_are_consistent() {
        let mut engine = engine();
        let applied = engine.hamiltonian_action(engine.amplitudes());
        let expectation = inner_product(engine.amplitudes(), &applied).re;
        assert!((expectation - engine.report().energy).abs() < 1.0e-12);
        let spectrum = engine.lanczos_spectrum(24);
        assert!(spectrum.ground_energy <= expectation + 1.0e-10);
        assert!(spectrum.first_excited_energy + 1.0e-10 >= spectrum.ground_energy);
        assert!(spectrum.gap >= 0.0);
        let cooled = engine.cool(600);
        assert!((cooled.energy - spectrum.ground_energy).abs() < 0.1);
    }

    #[test]
    fn periodic_pyrochlore_generator_scales_to_sixteen_spins() {
        for (dimensions, expected_spins) in [((2, 1, 1), 8), ((3, 1, 1), 12), ((2, 2, 1), 16)] {
            let (geometry, bonds) = periodic_pyrochlore_model(
                dimensions.0,
                dimensions.1,
                dimensions.2,
                SymmetryThermodynamicConfig::default(),
            )
            .unwrap();
            assert_eq!(geometry.vertices.len(), expected_spins);
            assert_eq!(geometry.tetrahedra.len(), expected_spins / 2);
            assert!(!geometry.edges.is_empty());
            assert_eq!(
                bonds
                    .iter()
                    .map(|bond| bond.multiplicity as usize)
                    .sum::<usize>(),
                3 * expected_spins
            );
            let mut degree = vec![0usize; expected_spins];
            for bond in bonds {
                degree[bond.a] += bond.multiplicity as usize;
                degree[bond.b] += bond.multiplicity as usize;
            }
            assert!(degree.into_iter().all(|degree| degree == 6));
        }
    }

    fn centroid(geometry: &SymmetryThermodynamicSubstrate) -> Vec3 {
        geometry
            .vertices
            .iter()
            .map(|vertex| vertex.position)
            .fold(Vec3::default(), |sum, position| sum + position)
            / geometry.vertices.len() as f64
    }
}
