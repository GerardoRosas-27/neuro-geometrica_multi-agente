//! Motor termodinámico fasorial independiente.
//!
//! Reutiliza la configuración y la topología del CDT nativo para construir un
//! Laplaciano magnético disperso, pero conserva su propio estado complejo y su
//! propia dinámica. No reemplaza ni modifica el motor CDT existente.

use crate::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use num_complex::Complex32;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fmt;

const EPSILON: f32 = 1.0e-7;
const PARALLEL_NODE_THRESHOLD: usize = 2_048;
const PARALLEL_EDGE_THRESHOLD: usize = 8_192;
pub const DEFAULT_PHASOR_STARTUP_SLICES: usize = 3;
pub const DEFAULT_PHASOR_NODES_PER_SLICE: usize = 65_536;
pub const DEFAULT_PHASOR_STARTUP_NODES: usize =
    DEFAULT_PHASOR_STARTUP_SLICES * DEFAULT_PHASOR_NODES_PER_SLICE;

#[derive(Clone, Copy, Debug)]
pub struct NativePhasorConfig {
    /// Intensidad del acoplamiento electromagnético discreto.
    pub coupling_strength: f32,
    /// Pozo radial que conserva amplitudes distintas de cero.
    pub radial_strength: f32,
    /// Amplitud de equilibrio del pozo radial.
    pub target_amplitude: f32,
    /// Confinamiento cuadrático adicional.
    pub confinement: f32,
    /// Peso del estímulo complejo externo.
    pub stimulus_gain: f32,
    /// Decaimiento del estímulo después de cada paso.
    pub stimulus_decay: f32,
    /// Peso de la entropía de amplitudes en F = U - T S.
    pub entropy_weight: f32,
    /// Escala global de temperatura y ruido.
    pub temperature_scale: f32,
    /// Paso del integrador de gradiente-Langevin.
    pub dt: f32,
    /// Escala del ruido térmico complejo.
    pub noise_scale: f32,
    /// Límite de amplitud para proteger la integración.
    pub max_amplitude: f32,
    /// Umbral usado para contar fasores activos.
    pub active_threshold: f32,
    pub seed: u64,
}

impl Default for NativePhasorConfig {
    fn default() -> Self {
        Self {
            coupling_strength: 1.0,
            radial_strength: 1.0,
            target_amplitude: 1.0,
            confinement: 0.02,
            stimulus_gain: 1.0,
            stimulus_decay: 0.92,
            entropy_weight: 1.0,
            temperature_scale: 0.05,
            dt: 0.02,
            noise_scale: 1.0,
            max_amplitude: 4.0,
            active_threshold: 1.0e-3,
            seed: 0xF450_12A5_0C1A_7701,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NativePhasorReport {
    pub tick: u64,
    pub nodes: usize,
    pub edges: usize,
    pub free_energy: f32,
    pub internal_energy: f32,
    pub coupling_energy: f32,
    pub radial_energy: f32,
    pub confinement_energy: f32,
    pub stimulus_energy: f32,
    pub entropy: f32,
    pub effective_temperature: f32,
    pub gradient_residual: f32,
    pub mean_amplitude: f32,
    pub phase_coherence: f32,
    pub active_phasors: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NativePhasorRelaxationReport {
    pub initial: NativePhasorReport,
    pub final_report: NativePhasorReport,
    pub best: NativePhasorReport,
    pub steps: usize,
    pub converged: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePhasorAnnealingSchedule {
    pub initial_temperature_scale: f32,
    pub final_temperature_scale: f32,
    pub steps: usize,
    pub energy_tolerance: f32,
    pub residual_tolerance: f32,
}

impl Default for NativePhasorAnnealingSchedule {
    fn default() -> Self {
        Self {
            initial_temperature_scale: 0.08,
            final_temperature_scale: 0.0,
            steps: 500,
            energy_tolerance: 1.0e-6,
            residual_tolerance: 1.0e-3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePhasorMinimizerConfig {
    pub max_iterations: usize,
    pub initial_step: f32,
    pub min_step: f32,
    pub max_step: f32,
    pub step_growth: f32,
    pub armijo_factor: f32,
    pub max_backtracks: usize,
    pub energy_tolerance: f32,
    pub residual_tolerance: f32,
    pub jacobi_preconditioner: bool,
    /// Sincronización de gauge O(E) antes del descenso continuo.
    pub topological_warm_start: bool,
}

impl Default for NativePhasorMinimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 400,
            initial_step: 1.0,
            min_step: 1.0e-6,
            max_step: 4.0,
            step_growth: 1.25,
            armijo_factor: 1.0e-4,
            max_backtracks: 12,
            energy_tolerance: 1.0e-7,
            residual_tolerance: 1.0e-4,
            jacobi_preconditioner: true,
            topological_warm_start: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NativePhasorMinimizationReport {
    pub initial: NativePhasorReport,
    pub final_report: NativePhasorReport,
    pub iterations: usize,
    pub energy_evaluations: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub warm_start_applied: bool,
    pub converged: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativePhasorError {
    EmptySubstrate,
    InvalidStateDimensions,
    InvalidEdge {
        edge: usize,
        node: usize,
        nodes: usize,
    },
}

impl fmt::Display for NativePhasorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySubstrate => write!(formatter, "el sustrato fasorial no contiene nodos"),
            Self::InvalidStateDimensions => {
                write!(
                    formatter,
                    "amplitud, fase y temperatura tienen dimensiones distintas"
                )
            }
            Self::InvalidEdge { edge, node, nodes } => write!(
                formatter,
                "la arista {edge} referencia el nodo {node}, pero sólo existen {nodes} nodos"
            ),
        }
    }
}

impl std::error::Error for NativePhasorError {}

#[derive(Clone, Copy, Debug)]
struct MagneticEdge {
    a: usize,
    b: usize,
    weight: f32,
    phase: f32,
}

#[derive(Clone, Debug)]
struct MagneticLaplacian {
    nodes: usize,
    row_offsets: Vec<usize>,
    columns: Vec<usize>,
    weights: Vec<f32>,
    transports: Vec<Complex32>,
    diagonal: Vec<f32>,
    edges: Vec<MagneticEdge>,
}

impl MagneticLaplacian {
    fn compile(core: &NativeThermoCdtSubstrate) -> Result<Self, NativePhasorError> {
        let nodes = core.node_count();
        if nodes == 0 {
            return Err(NativePhasorError::EmptySubstrate);
        }
        if core.amplitude.len() != nodes
            || core.phase.len() != nodes
            || core.temperature.len() != nodes
        {
            return Err(NativePhasorError::InvalidStateDimensions);
        }

        let edge_count = core
            .edge_a
            .len()
            .min(core.edge_b.len())
            .min(core.edge_weight.len())
            .min(core.edge_phase.len());
        let mut adjacency = vec![Vec::<(usize, f32, Complex32)>::new(); nodes];
        let mut diagonal = vec![0.0; nodes];
        let mut edges = Vec::with_capacity(edge_count);

        for edge in 0..edge_count {
            let a = core.edge_a[edge];
            let b = core.edge_b[edge];
            if a >= nodes {
                return Err(NativePhasorError::InvalidEdge {
                    edge,
                    node: a,
                    nodes,
                });
            }
            if b >= nodes {
                return Err(NativePhasorError::InvalidEdge {
                    edge,
                    node: b,
                    nodes,
                });
            }
            if a == b {
                continue;
            }
            let weight = core.edge_weight[edge].max(0.0);
            if weight <= EPSILON {
                continue;
            }
            let phase = core.edge_phase[edge];
            let forward_transport = Complex32::from_polar(1.0, -phase);
            adjacency[a].push((b, weight, forward_transport));
            adjacency[b].push((a, weight, forward_transport.conj()));
            diagonal[a] += weight;
            diagonal[b] += weight;
            edges.push(MagneticEdge {
                a,
                b,
                weight,
                phase,
            });
        }

        let mut row_offsets = Vec::with_capacity(nodes + 1);
        let mut columns = Vec::with_capacity(edges.len() * 2);
        let mut weights = Vec::with_capacity(edges.len() * 2);
        let mut transports = Vec::with_capacity(edges.len() * 2);
        row_offsets.push(0);
        for row in adjacency {
            for (column, weight, transport) in row {
                columns.push(column);
                weights.push(weight);
                transports.push(transport);
            }
            row_offsets.push(columns.len());
        }

        Ok(Self {
            nodes,
            row_offsets,
            columns,
            weights,
            transports,
            diagonal,
            edges,
        })
    }

    fn apply(&self, input: &[Complex32], output: &mut [Complex32]) {
        debug_assert_eq!(input.len(), self.nodes);
        debug_assert_eq!(output.len(), self.nodes);
        let apply_row = |row: usize| {
            let mut sum = input[row] * self.diagonal[row];
            for cursor in self.row_offsets[row]..self.row_offsets[row + 1] {
                sum -= input[self.columns[cursor]] * self.transports[cursor] * self.weights[cursor];
            }
            sum
        };
        if self.nodes < PARALLEL_NODE_THRESHOLD {
            for (row, value) in output.iter_mut().enumerate() {
                *value = apply_row(row);
            }
        } else {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(row, value)| *value = apply_row(row));
        }
    }

    fn coupling_energy(&self, phasors: &[Complex32], strength: f32) -> f32 {
        let edge_energy = |edge: &MagneticEdge| {
            let transport = Complex32::from_polar(1.0, -edge.phase);
            edge.weight * (phasors[edge.a] - transport * phasors[edge.b]).norm_sqr()
        };
        let sum = if self.edges.len() < PARALLEL_EDGE_THRESHOLD {
            self.edges.iter().map(edge_energy).sum::<f32>()
        } else {
            self.edges.par_iter().map(edge_energy).sum::<f32>()
        };
        0.5 * strength * sum
    }

    fn phase_coherence(&self, phasors: &[Complex32]) -> f32 {
        let (weighted_coherence, weight_sum) = self
            .edges
            .par_iter()
            .map(|edge| {
                let phase_delta = phasors[edge.a].arg() - phasors[edge.b].arg() + edge.phase;
                (edge.weight * phase_delta.cos(), edge.weight)
            })
            .reduce(
                || (0.0, 0.0),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if weight_sum <= EPSILON {
            1.0
        } else {
            (weighted_coherence / weight_sum).clamp(-1.0, 1.0)
        }
    }
}

/// Motor separado con estado complejo, Laplaciano magnético y dinámica
/// gradiente-Langevin. El CDT se usa únicamente como fuente de configuración,
/// topología y estado inicial.
#[derive(Clone, Debug)]
pub struct NativePhasorThermodynamicEngine {
    pub config: NativePhasorConfig,
    pub phasors: Vec<Complex32>,
    pub temperature: Vec<f32>,
    pub stimulus: Vec<Complex32>,
    operator: MagneticLaplacian,
    laplacian_buffer: Vec<Complex32>,
    tick: u64,
}

impl NativePhasorThermodynamicEngine {
    pub fn from_cdt_config(
        cdt_config: NativeThermoCdtConfig,
        phasor_config: NativePhasorConfig,
    ) -> Result<Self, NativePhasorError> {
        let core = NativeThermoCdtSubstrate::new(cdt_config);
        Self::from_core(&core, phasor_config)
    }

    pub fn from_core(
        core: &NativeThermoCdtSubstrate,
        config: NativePhasorConfig,
    ) -> Result<Self, NativePhasorError> {
        let operator = MagneticLaplacian::compile(core)?;
        let phasors = core
            .amplitude
            .iter()
            .copied()
            .zip(core.phase.iter().copied())
            .map(|(amplitude, phase)| Complex32::from_polar(amplitude.max(0.0), phase))
            .collect::<Vec<_>>();
        let nodes = phasors.len();
        Ok(Self {
            config: sanitize_config(config),
            phasors,
            temperature: core.temperature.clone(),
            stimulus: vec![Complex32::new(0.0, 0.0); nodes],
            operator,
            laplacian_buffer: vec![Complex32::new(0.0, 0.0); nodes],
            tick: 0,
        })
    }

    /// Recompila pesos y fases de arista desde el core CDT sin perder el
    /// atractor fasorial que está activo.
    pub fn recompile_from_core(
        &mut self,
        core: &NativeThermoCdtSubstrate,
    ) -> Result<(), NativePhasorError> {
        if core.node_count() != self.node_count() {
            return Err(NativePhasorError::InvalidStateDimensions);
        }
        self.operator = MagneticLaplacian::compile(core)?;
        self.temperature.copy_from_slice(&core.temperature);
        Ok(())
    }

    /// Reinicia el campo fasorial desde amplitudes y fases persistidas en CDT.
    pub fn synchronize_state_from_core(
        &mut self,
        core: &NativeThermoCdtSubstrate,
    ) -> Result<(), NativePhasorError> {
        self.recompile_from_core(core)?;
        for ((phasor, amplitude), phase) in self
            .phasors
            .iter_mut()
            .zip(&core.amplitude)
            .zip(&core.phase)
        {
            *phasor = Complex32::from_polar(amplitude.max(0.0), *phase);
        }
        Ok(())
    }

    pub fn node_count(&self) -> usize {
        self.phasors.len()
    }

    pub fn edge_count(&self) -> usize {
        self.operator.edges.len()
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn set_temperature_scale(&mut self, scale: f32) {
        self.config.temperature_scale = scale.max(0.0);
    }

    pub fn inject_pattern(&mut self, nodes: &[usize], amplitude: f32, phase: f32) {
        let field = Complex32::from_polar(amplitude.max(0.0), phase);
        for &node in nodes {
            if let Some(stimulus) = self.stimulus.get_mut(node) {
                *stimulus += field;
            }
        }
    }

    pub fn clear_stimulus(&mut self) {
        self.stimulus.fill(Complex32::new(0.0, 0.0));
    }

    pub fn dominant_nodes(&self, limit: usize) -> Vec<(usize, f32, f32)> {
        let mut nodes = self
            .phasors
            .iter()
            .enumerate()
            .map(|(node, phasor)| (node, phasor.norm(), phasor.arg()))
            .collect::<Vec<_>>();
        nodes.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        nodes.truncate(limit.min(nodes.len()));
        nodes
    }

    pub fn step(&mut self) -> NativePhasorReport {
        self.operator
            .apply(&self.phasors, &mut self.laplacian_buffer);
        let entropy_state = entropy_state(&self.phasors);
        let effective_temperature = self.effective_temperature();
        let dt = self.config.dt;
        let coupling = self.config.coupling_strength;
        let radial = self.config.radial_strength;
        let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
        let confinement = self.config.confinement;
        let stimulus_gain = self.config.stimulus_gain;
        let entropy_gain = self.config.entropy_weight * effective_temperature;
        let noise_scale = self.config.noise_scale;
        let max_amplitude = self.config.max_amplitude;
        let temperature_scale = self.config.temperature_scale;
        let noise_seed = self.config.seed ^ self.tick.rotate_left(29);
        let laplacian = &self.laplacian_buffer;
        let stimulus = &self.stimulus;
        let temperatures = &self.temperature;

        self.phasors
            .par_iter_mut()
            .enumerate()
            .for_each(|(node, phasor)| {
                let current = *phasor;
                let norm_sqr = current.norm_sqr();
                let entropy_gradient =
                    entropy_gradient(current, norm_sqr, entropy_state, entropy_gain);
                let gradient = coupling * laplacian[node]
                    + radial * (norm_sqr - target_sqr) * current
                    + confinement * current
                    - stimulus_gain * stimulus[node]
                    + entropy_gradient;
                let thermal_std = noise_scale
                    * (2.0 * temperatures[node].max(0.0) * temperature_scale * dt).sqrt();
                let noise = Complex32::new(
                    gaussian_from_counter(noise_seed, (node as u64) * 2),
                    gaussian_from_counter(noise_seed, (node as u64) * 2 + 1),
                ) * thermal_std;
                let mut next = current - gradient * dt + noise;
                let amplitude = next.norm();
                if amplitude > max_amplitude {
                    next *= max_amplitude / amplitude;
                }
                *phasor = next;
            });

        let decay = self.config.stimulus_decay;
        self.stimulus
            .par_iter_mut()
            .for_each(|stimulus| *stimulus *= decay);
        self.tick = self.tick.wrapping_add(1);
        self.report()
    }

    pub fn report(&self) -> NativePhasorReport {
        let entropy_state = entropy_state(&self.phasors);
        let effective_temperature = self.effective_temperature();
        let coupling_energy = self
            .operator
            .coupling_energy(&self.phasors, self.config.coupling_strength);
        let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
        let radial_energy = 0.25
            * self.config.radial_strength
            * self
                .phasors
                .par_iter()
                .map(|phasor| {
                    let delta = phasor.norm_sqr() - target_sqr;
                    delta * delta
                })
                .sum::<f32>();
        let confinement_energy = 0.5
            * self.config.confinement
            * self
                .phasors
                .par_iter()
                .map(Complex32::norm_sqr)
                .sum::<f32>();
        let stimulus_energy = -self.config.stimulus_gain
            * self
                .phasors
                .par_iter()
                .zip(self.stimulus.par_iter())
                .map(|(phasor, field)| (phasor.conj() * field).re)
                .sum::<f32>();
        let internal_energy =
            coupling_energy + radial_energy + confinement_energy + stimulus_energy;
        let free_energy = internal_energy
            - self.config.entropy_weight * effective_temperature * entropy_state.entropy;
        let mut laplacian = vec![Complex32::new(0.0, 0.0); self.node_count()];
        self.operator.apply(&self.phasors, &mut laplacian);
        let residual_sqr = self
            .phasors
            .par_iter()
            .enumerate()
            .map(|(node, phasor)| {
                let entropy_gradient = entropy_gradient(
                    *phasor,
                    phasor.norm_sqr(),
                    entropy_state,
                    self.config.entropy_weight * effective_temperature,
                );
                let gradient = self.config.coupling_strength * laplacian[node]
                    + self.config.radial_strength * (phasor.norm_sqr() - target_sqr) * *phasor
                    + self.config.confinement * *phasor
                    - self.config.stimulus_gain * self.stimulus[node]
                    + entropy_gradient;
                gradient.norm_sqr()
            })
            .sum::<f32>();
        let amplitude_sum = self
            .phasors
            .par_iter()
            .map(|phasor| phasor.norm())
            .sum::<f32>();
        let active_phasors = self
            .phasors
            .par_iter()
            .filter(|phasor| phasor.norm() > self.config.active_threshold)
            .count();

        NativePhasorReport {
            tick: self.tick,
            nodes: self.node_count(),
            edges: self.edge_count(),
            free_energy,
            internal_energy,
            coupling_energy,
            radial_energy,
            confinement_energy,
            stimulus_energy,
            entropy: entropy_state.entropy,
            effective_temperature,
            gradient_residual: (residual_sqr / self.node_count().max(1) as f32).sqrt(),
            mean_amplitude: amplitude_sum / self.node_count().max(1) as f32,
            phase_coherence: self.operator.phase_coherence(&self.phasors),
            active_phasors,
        }
    }

    pub fn run_until_stable(
        &mut self,
        max_steps: usize,
        energy_tolerance: f32,
        residual_tolerance: f32,
    ) -> NativePhasorRelaxationReport {
        let initial = self.report();
        let mut previous = initial;
        let mut best = initial;
        let mut steps = 0;
        let mut converged = false;

        for _ in 0..max_steps {
            let current = self.step();
            steps += 1;
            if current.free_energy < best.free_energy {
                best = current;
            }
            let energy_delta = (current.free_energy - previous.free_energy).abs();
            if energy_delta <= energy_tolerance.max(0.0)
                && current.gradient_residual <= residual_tolerance.max(0.0)
            {
                converged = true;
                previous = current;
                break;
            }
            previous = current;
        }

        NativePhasorRelaxationReport {
            initial,
            final_report: previous,
            best,
            steps,
            converged,
        }
    }

    pub fn anneal(
        &mut self,
        schedule: NativePhasorAnnealingSchedule,
    ) -> NativePhasorRelaxationReport {
        let initial = self.report();
        let mut previous = initial;
        let mut best = initial;
        let mut best_state = self.phasors.clone();
        let mut steps = 0;
        let mut converged = false;
        let total_steps = schedule.steps.max(1);

        for step in 0..total_steps {
            let progress = if total_steps == 1 {
                1.0
            } else {
                step as f32 / (total_steps - 1) as f32
            };
            self.set_temperature_scale(
                schedule.initial_temperature_scale
                    + progress
                        * (schedule.final_temperature_scale - schedule.initial_temperature_scale),
            );
            let current = self.step();
            steps += 1;
            if current.free_energy < best.free_energy {
                best = current;
                best_state.copy_from_slice(&self.phasors);
            }
            let energy_delta = (current.free_energy - previous.free_energy).abs();
            if progress >= 0.95
                && energy_delta <= schedule.energy_tolerance.max(0.0)
                && current.gradient_residual <= schedule.residual_tolerance.max(0.0)
            {
                converged = true;
                previous = current;
                break;
            }
            previous = current;
        }

        if best.free_energy < previous.free_energy {
            self.phasors.copy_from_slice(&best_state);
        }
        let final_report = self.report();
        NativePhasorRelaxationReport {
            initial,
            final_report,
            best,
            steps,
            converged,
        }
    }

    /// Minimiza directamente la energía libre mediante gradiente
    /// precondicionado y búsqueda de línea de Armijo. A diferencia de `step`,
    /// no inyecta ruido y sólo acepta movimientos que reducen F.
    pub fn minimize_free_energy(
        &mut self,
        config: NativePhasorMinimizerConfig,
    ) -> NativePhasorMinimizationReport {
        let config = sanitize_minimizer_config(config);
        let initial = self.report();
        let mut current_energy = initial.free_energy;
        let mut gradient = vec![Complex32::new(0.0, 0.0); self.node_count()];
        let mut direction = vec![Complex32::new(0.0, 0.0); self.node_count()];
        let mut candidate = self.phasors.clone();
        let mut step_size = config.initial_step;
        let mut iterations = 0;
        let mut energy_evaluations = 0;
        let mut accepted_steps = 0;
        let mut rejected_steps = 0;
        let mut warm_start_applied = false;
        let mut converged = false;

        if config.topological_warm_start {
            self.topological_phase_warm_start();
            let warm_energy = self.free_energy_for(&self.phasors);
            energy_evaluations += 1;
            if warm_energy <= current_energy {
                current_energy = warm_energy;
                candidate.copy_from_slice(&self.phasors);
                warm_start_applied = true;
            } else {
                self.phasors.copy_from_slice(&candidate);
            }
        }

        for _ in 0..config.max_iterations {
            self.free_energy_gradient(&self.phasors, &mut gradient);
            let residual_sum = if self.node_count() < PARALLEL_NODE_THRESHOLD {
                gradient.iter().map(|value| value.norm_sqr()).sum::<f32>()
            } else {
                gradient.par_iter().map(Complex32::norm_sqr).sum::<f32>()
            };
            let residual_sqr = residual_sum / self.node_count().max(1) as f32;
            let residual = residual_sqr.sqrt();
            if residual <= config.residual_tolerance {
                converged = true;
                break;
            }

            let coupling = self.config.coupling_strength;
            let radial = self.config.radial_strength;
            let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
            let confinement = self.config.confinement;
            let precondition = |node: usize| {
                let denominator = if config.jacobi_preconditioner {
                    (coupling * self.operator.diagonal[node]
                        + radial * (self.phasors[node].norm_sqr() + target_sqr)
                        + confinement)
                        .max(EPSILON)
                } else {
                    1.0
                };
                gradient[node] / denominator
            };
            if self.node_count() < PARALLEL_NODE_THRESHOLD {
                for (node, value) in direction.iter_mut().enumerate() {
                    *value = precondition(node);
                }
            } else {
                direction
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(node, value)| *value = precondition(node));
            }
            let directional_derivative = if self.node_count() < PARALLEL_NODE_THRESHOLD {
                gradient
                    .iter()
                    .zip(&direction)
                    .map(|(gradient, direction)| (gradient.conj() * direction).re)
                    .sum::<f32>()
            } else {
                gradient
                    .par_iter()
                    .zip(direction.par_iter())
                    .map(|(gradient, direction)| (gradient.conj() * direction).re)
                    .sum::<f32>()
            }
            .max(EPSILON);

            let mut trial_step = step_size;
            let mut accepted_energy = current_energy;
            let mut accepted = false;
            for _ in 0..config.max_backtracks {
                let propose = |node: usize| {
                    let mut next = self.phasors[node] - direction[node] * trial_step;
                    let amplitude = next.norm();
                    if amplitude > self.config.max_amplitude {
                        next *= self.config.max_amplitude / amplitude;
                    }
                    next
                };
                if self.node_count() < PARALLEL_NODE_THRESHOLD {
                    for (node, value) in candidate.iter_mut().enumerate() {
                        *value = propose(node);
                    }
                } else {
                    candidate
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(node, value)| *value = propose(node));
                }
                let trial_energy = self.free_energy_for(&candidate);
                energy_evaluations += 1;
                if trial_energy
                    <= current_energy - config.armijo_factor * trial_step * directional_derivative
                {
                    accepted = true;
                    accepted_energy = trial_energy;
                    break;
                }
                rejected_steps += 1;
                trial_step *= 0.5;
                if trial_step < config.min_step {
                    break;
                }
            }

            iterations += 1;
            if !accepted {
                break;
            }
            let energy_delta = (current_energy - accepted_energy).max(0.0);
            self.phasors.copy_from_slice(&candidate);
            self.tick = self.tick.wrapping_add(1);
            current_energy = accepted_energy;
            accepted_steps += 1;
            step_size = (trial_step * config.step_growth).min(config.max_step);
            if energy_delta <= config.energy_tolerance && residual <= config.residual_tolerance {
                converged = true;
                break;
            }
        }

        let final_report = self.report();
        if final_report.gradient_residual <= config.residual_tolerance {
            converged = true;
        }
        NativePhasorMinimizationReport {
            initial,
            final_report,
            iterations,
            energy_evaluations,
            accepted_steps,
            rejected_steps,
            warm_start_applied,
            converged,
        }
    }

    /// Mínimo global analítico para el caso no frustrado, sin estímulo y a
    /// temperatura efectiva cero. Sirve como oráculo de correctitud.
    pub fn analytic_unfrustrated_minimum(&self) -> Option<f32> {
        if self
            .operator
            .edges
            .iter()
            .any(|edge| edge.phase.sin().abs() > 1.0e-6 || 1.0 - edge.phase.cos() > 1.0e-6)
            || self.stimulus.iter().any(|field| field.norm_sqr() > EPSILON)
            || self.config.entropy_weight * self.effective_temperature() > EPSILON
            || self.config.radial_strength <= EPSILON
        {
            return None;
        }
        let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
        let equilibrium_sqr =
            (target_sqr - self.config.confinement / self.config.radial_strength).max(0.0);
        let radial_delta = equilibrium_sqr - target_sqr;
        let per_node = 0.25 * self.config.radial_strength * radial_delta * radial_delta
            + 0.5 * self.config.confinement * equilibrium_sqr;
        Some(per_node * self.node_count() as f32)
    }

    fn free_energy_gradient(&self, phasors: &[Complex32], output: &mut [Complex32]) {
        self.operator.apply(phasors, output);
        let entropy_state = entropy_state(phasors);
        let entropy_gain = self.config.entropy_weight * self.effective_temperature();
        let coupling = self.config.coupling_strength;
        let radial = self.config.radial_strength;
        let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
        let confinement = self.config.confinement;
        let stimulus_gain = self.config.stimulus_gain;
        let complete_gradient = |node: usize, laplacian: Complex32| {
            let phasor = phasors[node];
            coupling * laplacian
                + radial * (phasor.norm_sqr() - target_sqr) * phasor
                + confinement * phasor
                - stimulus_gain * self.stimulus[node]
                + entropy_gradient(phasor, phasor.norm_sqr(), entropy_state, entropy_gain)
        };
        if output.len() < PARALLEL_NODE_THRESHOLD {
            for (node, value) in output.iter_mut().enumerate() {
                *value = complete_gradient(node, *value);
            }
        } else {
            output.par_iter_mut().enumerate().for_each(|(node, value)| {
                *value = complete_gradient(node, *value);
            });
        }
    }

    fn topological_phase_warm_start(&mut self) {
        let nodes = self.node_count();
        let mut synchronized = vec![Complex32::new(0.0, 0.0); nodes];
        let mut visited = vec![false; nodes];
        let mut queue = VecDeque::new();
        for root in 0..nodes {
            if visited[root] {
                continue;
            }
            let root_norm = self.phasors[root].norm();
            synchronized[root] = if root_norm > EPSILON {
                self.phasors[root] / root_norm
            } else {
                Complex32::new(1.0, 0.0)
            };
            visited[root] = true;
            queue.push_back(root);
            while let Some(node) = queue.pop_front() {
                for cursor in self.operator.row_offsets[node]..self.operator.row_offsets[node + 1] {
                    let neighbor = self.operator.columns[cursor];
                    if visited[neighbor] {
                        continue;
                    }
                    synchronized[neighbor] =
                        self.operator.transports[cursor].conj() * synchronized[node];
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        let can_use_radial_equilibrium = self
            .stimulus
            .iter()
            .all(|field| field.norm_sqr() <= EPSILON)
            && self.config.entropy_weight * self.effective_temperature() <= EPSILON
            && self.config.radial_strength > EPSILON;
        let equilibrium_amplitude = if can_use_radial_equilibrium {
            let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
            (target_sqr - self.config.confinement / self.config.radial_strength)
                .max(0.0)
                .sqrt()
        } else {
            0.0
        };
        self.phasors
            .par_iter_mut()
            .enumerate()
            .for_each(|(node, phasor)| {
                let amplitude = if can_use_radial_equilibrium {
                    equilibrium_amplitude
                } else {
                    phasor.norm()
                };
                *phasor = synchronized[node] * amplitude;
            });
    }

    fn free_energy_for(&self, phasors: &[Complex32]) -> f32 {
        let coupling = self
            .operator
            .coupling_energy(phasors, self.config.coupling_strength);
        let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
        let local_energy = |(phasor, field): (&Complex32, &Complex32)| {
            let radial_delta = phasor.norm_sqr() - target_sqr;
            0.25 * self.config.radial_strength * radial_delta * radial_delta
                + 0.5 * self.config.confinement * phasor.norm_sqr()
                - self.config.stimulus_gain * (phasor.conj() * field).re
        };
        let local = if phasors.len() < PARALLEL_NODE_THRESHOLD {
            phasors
                .iter()
                .zip(&self.stimulus)
                .map(local_energy)
                .sum::<f32>()
        } else {
            phasors
                .par_iter()
                .zip(self.stimulus.par_iter())
                .map(local_energy)
                .sum::<f32>()
        };
        coupling + local
            - self.config.entropy_weight
                * self.effective_temperature()
                * entropy_state(phasors).entropy
    }

    fn effective_temperature(&self) -> f32 {
        let mean_temperature =
            self.temperature.iter().copied().sum::<f32>() / self.temperature.len().max(1) as f32;
        mean_temperature.max(0.0) * self.config.temperature_scale
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EntropyState {
    entropy: f32,
    q_sum: f32,
    q_log_q_sum: f32,
}

fn entropy_state(phasors: &[Complex32]) -> EntropyState {
    let contribution = |phasor: &Complex32| {
        let q = phasor.norm_sqr() + EPSILON;
        (q, q * q.ln())
    };
    let (q_sum, q_log_q_sum) = if phasors.len() < PARALLEL_NODE_THRESHOLD {
        phasors
            .iter()
            .map(contribution)
            .fold((0.0, 0.0), |left, right| {
                (left.0 + right.0, left.1 + right.1)
            })
    } else {
        phasors.par_iter().map(contribution).reduce(
            || (0.0, 0.0),
            |left, right| (left.0 + right.0, left.1 + right.1),
        )
    };
    if q_sum <= EPSILON {
        return EntropyState::default();
    }
    EntropyState {
        entropy: (q_sum.ln() - q_log_q_sum / q_sum).max(0.0),
        q_sum,
        q_log_q_sum,
    }
}

fn entropy_gradient(
    phasor: Complex32,
    norm_sqr: f32,
    state: EntropyState,
    entropy_gain: f32,
) -> Complex32 {
    if entropy_gain <= EPSILON || state.q_sum <= EPSILON {
        return Complex32::new(0.0, 0.0);
    }
    let q = norm_sqr + EPSILON;
    let d_entropy_d_q = (state.q_log_q_sum / state.q_sum - q.ln()) / state.q_sum;
    phasor * (-2.0 * entropy_gain * d_entropy_d_q)
}

fn sanitize_config(config: NativePhasorConfig) -> NativePhasorConfig {
    NativePhasorConfig {
        coupling_strength: config.coupling_strength.max(0.0),
        radial_strength: config.radial_strength.max(0.0),
        target_amplitude: config.target_amplitude.max(EPSILON),
        confinement: config.confinement.max(0.0),
        stimulus_gain: config.stimulus_gain.max(0.0),
        stimulus_decay: config.stimulus_decay.clamp(0.0, 1.0),
        entropy_weight: config.entropy_weight.max(0.0),
        temperature_scale: config.temperature_scale.max(0.0),
        dt: config.dt.clamp(EPSILON, 0.25),
        noise_scale: config.noise_scale.max(0.0),
        max_amplitude: config
            .max_amplitude
            .max(config.target_amplitude.max(EPSILON)),
        active_threshold: config.active_threshold.max(0.0),
        ..config
    }
}

fn sanitize_minimizer_config(config: NativePhasorMinimizerConfig) -> NativePhasorMinimizerConfig {
    NativePhasorMinimizerConfig {
        max_iterations: config.max_iterations.max(1),
        initial_step: config.initial_step.max(EPSILON),
        min_step: config
            .min_step
            .clamp(EPSILON, config.initial_step.max(EPSILON)),
        max_step: config.max_step.max(config.initial_step.max(EPSILON)),
        step_growth: config.step_growth.max(1.0),
        armijo_factor: config.armijo_factor.clamp(EPSILON, 0.5),
        max_backtracks: config.max_backtracks.max(1),
        energy_tolerance: config.energy_tolerance.max(0.0),
        residual_tolerance: config.residual_tolerance.max(0.0),
        ..config
    }
}

#[inline(always)]
fn gaussian_from_counter(seed: u64, counter: u64) -> f32 {
    let base = seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    unit_from_u64(splitmix64(base))
        + unit_from_u64(splitmix64(base ^ 0xA24B_AED4_963E_E407))
        + unit_from_u64(splitmix64(base ^ 0x9FB2_1C65_1E98_DF25))
        + unit_from_u64(splitmix64(base ^ 0xC13F_A9A9_02A6_328F))
        + unit_from_u64(splitmix64(base ^ 0x91E1_0DA5_C79E_7B1D))
        + unit_from_u64(splitmix64(base ^ 0xD1B5_4A32_D192_ED03))
        - 3.0
}

#[inline(always)]
fn unit_from_u64(value: u64) -> f32 {
    ((value >> 40) as f32) * (1.0 / (1_u32 << 24) as f32)
}

#[inline(always)]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_thermodynamic_cdt::NativeCdtEdgeKind;

    fn two_node_core(edge_phase: f32) -> NativeThermoCdtSubstrate {
        let mut core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: 2,
            spatial_degree: 1,
            temporal_degree: 1,
            temperature: 0.0,
            ..NativeThermoCdtConfig::default()
        });
        core.replace_edges([(0, 1, NativeCdtEdgeKind::Spatial, 1.0, edge_phase, 1.0)]);
        core.amplitude.fill(1.0);
        core.phase.fill(0.0);
        core
    }

    #[test]
    fn aligned_phasors_have_zero_magnetic_coupling_energy() {
        let core = two_node_core(0.0);
        let engine =
            NativePhasorThermodynamicEngine::from_core(&core, NativePhasorConfig::default())
                .unwrap();
        assert!(engine.report().coupling_energy.abs() < 1.0e-7);
    }

    #[test]
    fn magnetic_phase_offset_defines_the_zero_energy_alignment() {
        let offset = 0.7;
        let core = two_node_core(offset);
        let mut engine =
            NativePhasorThermodynamicEngine::from_core(&core, NativePhasorConfig::default())
                .unwrap();
        engine.phasors[0] = Complex32::from_polar(1.0, -offset);
        engine.phasors[1] = Complex32::from_polar(1.0, 0.0);
        assert!(engine.report().coupling_energy.abs() < 1.0e-6);
    }

    #[test]
    fn edge_sum_matches_the_sparse_quadratic_form() {
        let core = two_node_core(0.35);
        let mut engine =
            NativePhasorThermodynamicEngine::from_core(&core, NativePhasorConfig::default())
                .unwrap();
        engine.phasors[0] = Complex32::new(0.4, -0.8);
        engine.phasors[1] = Complex32::new(-0.2, 1.1);
        let mut action = vec![Complex32::new(0.0, 0.0); engine.node_count()];
        engine.operator.apply(&engine.phasors, &mut action);
        let quadratic = 0.5
            * engine
                .phasors
                .iter()
                .zip(&action)
                .map(|(phasor, value)| (phasor.conj() * value).re)
                .sum::<f32>();
        let edge_sum = engine.operator.coupling_energy(&engine.phasors, 1.0);
        assert!((quadratic - edge_sum).abs() < 1.0e-6);
    }

    #[test]
    fn deterministic_relaxation_lowers_free_energy() {
        let core = two_node_core(0.0);
        let mut engine = NativePhasorThermodynamicEngine::from_core(
            &core,
            NativePhasorConfig {
                temperature_scale: 0.0,
                noise_scale: 0.0,
                entropy_weight: 0.0,
                dt: 0.03,
                ..NativePhasorConfig::default()
            },
        )
        .unwrap();
        engine.phasors[0] = Complex32::from_polar(1.0, 0.0);
        engine.phasors[1] = Complex32::from_polar(1.0, std::f32::consts::PI * 0.8);
        let before = engine.report();
        let relaxation = engine.run_until_stable(400, 1.0e-7, 2.0e-3);
        assert!(
            relaxation.final_report.free_energy < before.free_energy,
            "{relaxation:?}"
        );
        assert!(
            relaxation.final_report.phase_coherence > before.phase_coherence,
            "{relaxation:?}"
        );
    }

    #[test]
    fn radial_potential_prevents_zero_amplitude_collapse() {
        let core = two_node_core(0.0);
        let mut engine = NativePhasorThermodynamicEngine::from_core(
            &core,
            NativePhasorConfig {
                temperature_scale: 0.0,
                noise_scale: 0.0,
                entropy_weight: 0.0,
                confinement: 0.0,
                dt: 0.05,
                ..NativePhasorConfig::default()
            },
        )
        .unwrap();
        engine.phasors.fill(Complex32::new(0.1, 0.0));
        engine.run_until_stable(300, 1.0e-7, 2.0e-3);
        let report = engine.report();
        assert!(report.mean_amplitude > 0.9, "{report:?}");
        assert_eq!(report.active_phasors, 2);
    }

    #[test]
    fn preconditioned_minimizer_reaches_the_known_global_minimum_efficiently() {
        let core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: 128,
            spatial_degree: 1,
            temporal_degree: 1,
            temperature: 0.0,
            seed: 44_071,
            ..NativeThermoCdtConfig::default()
        });
        let mut engine = NativePhasorThermodynamicEngine::from_core(
            &core,
            NativePhasorConfig {
                coupling_strength: 1.0,
                radial_strength: 1.0,
                target_amplitude: 1.0,
                confinement: 0.02,
                entropy_weight: 0.0,
                temperature_scale: 0.0,
                noise_scale: 0.0,
                ..NativePhasorConfig::default()
            },
        )
        .unwrap();
        let exact_minimum = engine.analytic_unfrustrated_minimum().unwrap();
        let mut fixed_step_baseline = engine.clone();
        let fixed_step_result = fixed_step_baseline.run_until_stable(300, 1.0e-7, 2.0e-4);
        let fixed_step_gap = (fixed_step_result.final_report.free_energy - exact_minimum).abs();
        let result = engine.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: 300,
            residual_tolerance: 2.0e-4,
            ..NativePhasorMinimizerConfig::default()
        });
        let relative_gap =
            (result.final_report.free_energy - exact_minimum).abs() / exact_minimum.abs().max(1.0);

        assert!(result.converged, "{result:?}");
        assert!(relative_gap < 2.0e-5, "gap={relative_gap} {result:?}");
        assert!(result.final_report.phase_coherence > 0.999_98, "{result:?}");
        assert!(result.warm_start_applied, "{result:?}");
        assert!(result.iterations <= 1, "{result:?}");
        assert!(
            (result.final_report.free_energy - exact_minimum).abs() * 1_000.0 < fixed_step_gap,
            "optimizado={result:?} baseline={fixed_step_result:?}"
        );
        assert!(
            result.energy_evaluations <= result.iterations * 4 + 1,
            "{result:?}"
        );
    }

    #[test]
    fn new_engine_does_not_mutate_the_source_core() {
        let core = two_node_core(0.0);
        let original_amplitude = core.amplitude.clone();
        let original_phase = core.phase.clone();
        let mut engine =
            NativePhasorThermodynamicEngine::from_core(&core, NativePhasorConfig::default())
                .unwrap();
        engine.inject_pattern(&[0], 2.0, 1.2);
        engine.step();
        assert_eq!(core.amplitude, original_amplitude);
        assert_eq!(core.phase, original_phase);
    }
}
