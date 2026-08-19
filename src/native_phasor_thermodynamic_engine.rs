//! Motor termodinámico fasorial independiente.
//!
//! Reutiliza la configuración y la topología del CDT nativo para construir un
//! Laplaciano magnético disperso, pero conserva su propio estado complejo y su
//! propia dinámica. No reemplaza ni modifica el motor CDT existente.

use crate::native_rng::{gaussian_from_counter, splitmix64, unit_from_u64};
use crate::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use num_complex::Complex32;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fmt;

const EPSILON: f32 = 1.0e-7;
/// Tamaños por debajo de los cuales el recorrido secuencial gana al reparto en
/// `rayon`. Cada iteración del minimizador abre varias regiones paralelas muy
/// cortas, así que repartir un vector pequeño cuesta más que recorrerlo entero:
/// medido sobre el solver de Armijo, 2 048 nodos tardaban 23 ms en paralelo
/// frente a 6 ms en serie. El cruce real está entre 8 192 y 16 384 nodos.
const PARALLEL_NODE_THRESHOLD: usize = 16_384;
const PARALLEL_EDGE_THRESHOLD: usize = 32_768;
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

/// Cómo se programan los moduladores de dirección sobre el descenso Armijo.
///
/// Ninguno de los dos modos altera F ni el criterio de aceptación: sólo
/// deciden en qué iteraciones se reorienta la dirección antes de la búsqueda
/// de línea.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NativePhasorInferencePolicy {
    /// Aplica los moduladores configurados en todas las iteraciones.
    Fixed,
    /// Mantiene Handshake mientras la frontera siga desalineada, sondea Φ con
    /// histéresis y libera el descenso a Armijo puro cuando el residuo ya
    /// domina el progreso.
    #[default]
    Adaptive,
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
    /// Intensidad del precondicionador atencional sobre el residuo energético.
    /// Cero conserva exactamente el minimizador de energía libre.
    pub attention_strength: f32,
    /// Temperatura del softmax aplicado a `ln(1 + |gradiente|)`.
    pub attention_temperature: f32,
    /// Límite de ganancia para impedir que un único fasor domine el descenso.
    pub attention_max_gain: f32,
    /// Umbral de ignición Φ. Φ combina concentración atencional y coherencia
    /// de fase global; cero mantiene la atención suave siempre activa.
    pub attention_ignition_threshold: f32,
    /// Intensidad del precondicionador Handshake. Usa el mismo estímulo que ya
    /// forma parte de F como frontera final; cero conserva Armijo exactamente.
    pub handshake_strength: f32,
    /// Rondas de propagación adjunta de la frontera sobre el grafo magnético.
    pub handshake_rounds: usize,
    /// Peso residual de la frontera anterior en cada ronda de mensajes.
    pub handshake_damping: f32,
    /// Límite de ganancia local del precondicionador Handshake.
    pub handshake_max_gain: f32,
    /// Programación de los moduladores sobre el descenso.
    pub inference_policy: NativePhasorInferencePolicy,
    /// Fracción del residuo inicial por debajo de la cual `Adaptive` libera
    /// los moduladores: cerca del mínimo, reorientar sólo estorba al residuo.
    pub modifier_release_residual_ratio: f32,
    /// Coherencia media con la frontera a partir de la cual Handshake ya no
    /// reorienta nada y se apaga para el resto del descenso.
    pub handshake_saturation_coherence: f32,
    /// Iteraciones entre sondas de Φ mientras la atención está apagada.
    pub attention_probe_interval: usize,
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
            attention_strength: 0.0,
            attention_temperature: 1.0,
            attention_max_gain: 4.0,
            attention_ignition_threshold: 0.0,
            handshake_strength: 0.0,
            handshake_rounds: 4,
            handshake_damping: 0.25,
            handshake_max_gain: 4.0,
            inference_policy: NativePhasorInferencePolicy::Adaptive,
            modifier_release_residual_ratio: 0.25,
            handshake_saturation_coherence: 0.98,
            attention_probe_interval: 4,
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
    /// Entropía normalizada media de la atención (1 = uniforme).
    pub mean_attention_entropy: f32,
    /// Mayor ganancia atencional aplicada a un fasor.
    pub peak_attention_gain: f32,
    /// Φ medio: foco no uniforme multiplicado por integración global de fase.
    pub mean_integrated_information: f32,
    /// Iteraciones en las que Φ alcanzó el umbral de ignición.
    pub attention_ignitions: usize,
    /// Indica que existió una frontera no nula y se aplicó Handshake.
    pub handshake_applied: bool,
    /// Productos dispersos usados para propagar la frontera adjunta.
    pub handshake_operator_applications: usize,
    /// Coherencia media entre estado forward y frontera backward.
    pub mean_handshake_coherence: f32,
    /// Iteraciones en las que Handshake reorientó la dirección.
    pub handshake_iterations: usize,
    /// Iteraciones en las que se evaluó Φ.
    pub attention_probes: usize,
    /// Iteración en la que `Adaptive` liberó los moduladores. Igual a
    /// `iterations` cuando nunca se liberaron.
    pub modifier_release_iteration: usize,
}

/// Configuración experimental del bucle local de inferencia activa.
///
/// Cada barrido combina Metropolis-within-Gibbs sobre un único fasor con un
/// descenso coordenado opcional. Ambas operaciones sólo consultan las aristas
/// incidentes al nodo actualizado. No es la ruta de producción: los benchmarks
/// seleccionan `minimize_free_energy` por menor tiempo y menor energía final.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePhasorActiveInferenceConfig {
    pub sweeps: usize,
    pub burn_in_sweeps: usize,
    pub sampling_temperature: f32,
    pub proposal_std: f32,
    pub local_learning_rate: f32,
    pub entropy_samples: usize,
    pub keep_best: bool,
    pub seed: u64,
}

impl Default for NativePhasorActiveInferenceConfig {
    fn default() -> Self {
        Self {
            sweeps: 200,
            burn_in_sweeps: 40,
            sampling_temperature: 0.05,
            proposal_std: 0.20,
            local_learning_rate: 0.35,
            entropy_samples: 512,
            keep_best: true,
            seed: 0xAC71_1EFE_2026,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NativePhasorActiveInferenceReport {
    pub initial: NativePhasorReport,
    pub final_report: NativePhasorReport,
    pub best_free_energy: f32,
    pub sweeps: usize,
    pub gibbs_proposals: usize,
    pub gibbs_accepted: usize,
    pub local_updates_accepted: usize,
    pub sampled_mean_internal_energy: f32,
    pub sampled_entropy: f32,
    pub entropy_absolute_error: f32,
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
    /// `exp(-i * phase)` precalculado: evita un `sin`/`cos` por arista en cada
    /// evaluación de energía dentro de la búsqueda de línea.
    transport: Complex32,
}

#[derive(Clone, Debug)]
struct MagneticLaplacian {
    nodes: usize,
    row_offsets: Vec<usize>,
    columns: Vec<usize>,
    weights: Vec<f32>,
    transports: Vec<Complex32>,
    /// `transport * weight` por elemento no nulo. Colapsa la multiplicación
    /// compleja-por-escalar del producto disperso en una sola lectura.
    weighted_transports: Vec<Complex32>,
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
                transport: forward_transport,
            });
        }

        let mut row_offsets = Vec::with_capacity(nodes + 1);
        let mut columns = Vec::with_capacity(edges.len() * 2);
        let mut weights = Vec::with_capacity(edges.len() * 2);
        let mut transports = Vec::with_capacity(edges.len() * 2);
        let mut weighted_transports = Vec::with_capacity(edges.len() * 2);
        row_offsets.push(0);
        for row in adjacency {
            for (column, weight, transport) in row {
                columns.push(column);
                weights.push(weight);
                transports.push(transport);
                weighted_transports.push(transport * weight);
            }
            row_offsets.push(columns.len());
        }

        Ok(Self {
            nodes,
            row_offsets,
            columns,
            weights,
            transports,
            weighted_transports,
            diagonal,
            edges,
        })
    }

    fn apply(&self, input: &[Complex32], output: &mut [Complex32]) {
        debug_assert_eq!(input.len(), self.nodes);
        debug_assert_eq!(output.len(), self.nodes);
        let apply_row = |row: usize| {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            let columns = &self.columns[start..end];
            let transports = &self.weighted_transports[start..end];
            let mut sum = input[row] * self.diagonal[row];
            for (column, transport) in columns.iter().zip(transports) {
                sum -= input[*column] * *transport;
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

    /// Propaga mensajes desde una frontera por la adyacencia magnética
    /// normalizada. El transporte inverso ya está representado por el
    /// conjugado compilado en la fila opuesta; no implica inversión temporal.
    fn propagate_adjoint(&self, input: &[Complex32], output: &mut [Complex32]) {
        debug_assert_eq!(input.len(), self.nodes);
        debug_assert_eq!(output.len(), self.nodes);
        let propagate_row = |row: usize| {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            let denominator = self.diagonal[row].max(EPSILON);
            let mut sum = Complex32::new(0.0, 0.0);
            for cursor in start..end {
                sum += input[self.columns[cursor]] * self.weighted_transports[cursor];
            }
            sum / denominator
        };
        if self.nodes < PARALLEL_NODE_THRESHOLD {
            for (row, value) in output.iter_mut().enumerate() {
                *value = propagate_row(row);
            }
        } else {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(row, value)| *value = propagate_row(row));
        }
    }

    fn coupling_energy(&self, phasors: &[Complex32], strength: f32) -> f32 {
        let edge_energy = |edge: &MagneticEdge| {
            f64::from(edge.weight * (phasors[edge.a] - edge.transport * phasors[edge.b]).norm_sqr())
        };
        let sum = if self.edges.len() < PARALLEL_EDGE_THRESHOLD {
            self.edges.iter().map(edge_energy).sum::<f64>()
        } else {
            self.edges.par_iter().map(edge_energy).sum::<f64>()
        };
        0.5 * strength * sum as f32
    }

    /// Coherencia de fase sin trigonometría por arista.
    ///
    /// `cos(arg(a) - arg(b) + phase)` equivale a
    /// `Re(a * conj(b * transport)) / (|a| |b|)` porque `transport` es
    /// `exp(-i*phase)` y tiene módulo unitario. Sólo se recurre a `atan2`/`cos`
    /// cuando alguno de los dos módulos se anula y el cociente es indefinido.
    fn phase_coherence(&self, phasors: &[Complex32]) -> f32 {
        let edge_coherence = |edge: &MagneticEdge| {
            let left = phasors[edge.a];
            let right = phasors[edge.b];
            let magnitude = left.norm() * right.norm();
            let cosine = if magnitude > EPSILON {
                ((left * (edge.transport * right).conj()).re / magnitude).clamp(-1.0, 1.0)
            } else {
                (left.arg() - right.arg() + edge.phase).cos()
            };
            (f64::from(edge.weight * cosine), f64::from(edge.weight))
        };
        let combine = |left: (f64, f64), right: (f64, f64)| (left.0 + right.0, left.1 + right.1);
        let (weighted_coherence, weight_sum) = if self.edges.len() < PARALLEL_EDGE_THRESHOLD {
            self.edges
                .iter()
                .map(edge_coherence)
                .fold((0.0, 0.0), combine)
        } else {
            self.edges
                .par_iter()
                .map(edge_coherence)
                .reduce(|| (0.0, 0.0), combine)
        };
        if weight_sum <= f64::from(EPSILON) {
            1.0
        } else {
            (weighted_coherence / weight_sum).clamp(-1.0, 1.0) as f32
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
    /// Temperatura por nodo. Si se escribe directamente hay que llamar a
    /// [`Self::refresh_temperature_statistics`] para revalidar la media
    /// cacheada que consume `effective_temperature`.
    pub temperature: Vec<f32>,
    pub stimulus: Vec<Complex32>,
    operator: MagneticLaplacian,
    laplacian_buffer: Vec<Complex32>,
    minimizer_scratch: MinimizerScratch,
    /// Media de `temperature`. La búsqueda de línea evalúa la energía libre
    /// varias veces por iteración y cada evaluación necesitaba recorrer todo
    /// el vector de temperaturas para obtener este único escalar.
    mean_temperature: f32,
    tick: u64,
}

/// Buffers reutilizables del minimizador: `minimize_free_energy` es la ruta
/// caliente de inferencia (y se llama por candidato durante sleep), así que
/// gradiente, dirección y candidato se conservan entre llamadas en vez de
/// reservar 3×N complejos en cada invocación.
#[derive(Clone, Debug, Default)]
struct MinimizerScratch {
    gradient: Vec<Complex32>,
    direction: Vec<Complex32>,
    candidate: Vec<Complex32>,
    attention: Vec<f32>,
    /// Frontera adjunta ya normalizada por nodo: guarda la fase objetivo con
    /// módulo unitario para que la coherencia por iteración sea un producto
    /// escalar sin raíces cuadradas.
    handshake_boundary: Vec<Complex32>,
    handshake_buffer: Vec<Complex32>,
    /// `|frontera| / max|frontera|` precalculado: es estático durante todo el
    /// descenso y sacarlo del bucle evita dos recorridos por iteración.
    handshake_reachability: Vec<f32>,
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
        let temperature = core.temperature.clone();
        let mean_temperature = mean_temperature(&temperature);
        Ok(Self {
            config: sanitize_config(config),
            phasors,
            temperature,
            stimulus: vec![Complex32::new(0.0, 0.0); nodes],
            operator,
            laplacian_buffer: vec![Complex32::new(0.0, 0.0); nodes],
            minimizer_scratch: MinimizerScratch::default(),
            mean_temperature,
            tick: 0,
        })
    }

    /// Recalcula la media de temperatura cacheada. Necesario sólo si se escribe
    /// `temperature` sin pasar por `from_core` o `recompile_from_core`.
    pub fn refresh_temperature_statistics(&mut self) {
        self.mean_temperature = mean_temperature(&self.temperature);
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
        self.refresh_temperature_statistics();
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

    pub fn restore_runtime_state(
        &mut self,
        phasors: Vec<Complex32>,
        temperature: Vec<f32>,
        stimulus: Vec<Complex32>,
        tick: u64,
    ) -> Result<(), NativePhasorError> {
        if phasors.len() != self.phasors.len()
            || temperature.len() != self.temperature.len()
            || stimulus.len() != self.stimulus.len()
        {
            return Err(NativePhasorError::InvalidStateDimensions);
        }
        self.phasors = phasors;
        self.temperature = temperature;
        self.stimulus = stimulus;
        self.tick = tick;
        self.refresh_temperature_statistics();
        Ok(())
    }

    /// Restaura el contador de ticks tras una sonda o rollback que ejecutó
    /// pasos sólo para medir y debe dejar el motor como estaba.
    pub(crate) fn set_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Libera el workspace persistente del minimizador. Útil para medir la
    /// ruta fría o devolver memoria en motores que no volverán a inferir.
    pub fn clear_minimizer_workspace(&mut self) {
        self.minimizer_scratch = MinimizerScratch::default();
    }

    /// Capacidad reservada por los buffers del minimizador, en bytes.
    pub fn minimizer_workspace_capacity_bytes(&self) -> usize {
        (self.minimizer_scratch.gradient.capacity()
            + self.minimizer_scratch.direction.capacity()
            + self.minimizer_scratch.candidate.capacity()
            + self.minimizer_scratch.handshake_boundary.capacity()
            + self.minimizer_scratch.handshake_buffer.capacity())
            * std::mem::size_of::<Complex32>()
            + (self.minimizer_scratch.attention.capacity()
                + self.minimizer_scratch.handshake_reachability.capacity())
                * std::mem::size_of::<f32>()
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

        let max_amplitude_sqr = max_amplitude * max_amplitude;
        let integrate = |node: usize, phasor: &mut Complex32| {
            let current = *phasor;
            let norm_sqr = current.norm_sqr();
            let entropy_gradient = entropy_gradient(current, norm_sqr, entropy_state, entropy_gain);
            let gradient = coupling * laplacian[node]
                + radial * (norm_sqr - target_sqr) * current
                + confinement * current
                - stimulus_gain * stimulus[node]
                + entropy_gradient;
            let thermal_std =
                noise_scale * (2.0 * temperatures[node].max(0.0) * temperature_scale * dt).sqrt();
            let noise = Complex32::new(
                gaussian_from_counter(noise_seed, (node as u64) * 2),
                gaussian_from_counter(noise_seed, (node as u64) * 2 + 1),
            ) * thermal_std;
            let mut next = current - gradient * dt + noise;
            let amplitude_sqr = next.norm_sqr();
            if amplitude_sqr > max_amplitude_sqr {
                next *= max_amplitude / amplitude_sqr.sqrt();
            }
            *phasor = next;
        };
        let decay = self.config.stimulus_decay;
        if self.phasors.len() < PARALLEL_NODE_THRESHOLD {
            self.phasors
                .iter_mut()
                .enumerate()
                .for_each(|(node, phasor)| integrate(node, phasor));
            self.stimulus
                .iter_mut()
                .for_each(|stimulus| *stimulus *= decay);
        } else {
            self.phasors
                .par_iter_mut()
                .enumerate()
                .for_each(|(node, phasor)| integrate(node, phasor));
            self.stimulus
                .par_iter_mut()
                .for_each(|stimulus| *stimulus *= decay);
        }
        self.tick = self.tick.wrapping_add(1);
        let mut scratch = std::mem::take(&mut self.laplacian_buffer);
        let report = self.report_with_scratch(&mut scratch);
        self.laplacian_buffer = scratch;
        report
    }

    pub fn report(&self) -> NativePhasorReport {
        let mut scratch = vec![Complex32::new(0.0, 0.0); self.node_count()];
        self.report_with_scratch(&mut scratch)
    }

    /// Igual que [`Self::report`] pero reutilizando un buffer del llamador para
    /// el producto Laplaciano. Todas las métricas por nodo se acumulan en un
    /// solo recorrido en lugar de seis independientes.
    fn report_with_scratch(&self, laplacian: &mut [Complex32]) -> NativePhasorReport {
        let entropy_state = entropy_state(&self.phasors);
        let effective_temperature = self.effective_temperature();
        let entropy_gain = self.config.entropy_weight * effective_temperature;
        let coupling_energy = self
            .operator
            .coupling_energy(&self.phasors, self.config.coupling_strength);
        self.operator.apply(&self.phasors, laplacian);

        let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
        let coupling = self.config.coupling_strength;
        let radial = self.config.radial_strength;
        let confinement = self.config.confinement;
        let stimulus_gain = self.config.stimulus_gain;
        let active_threshold = self.config.active_threshold;
        let contribution =
            |((phasor, field), laplacian): ((&Complex32, &Complex32), &Complex32)| {
                let phasor = *phasor;
                let norm_sqr = phasor.norm_sqr();
                let radial_delta = norm_sqr - target_sqr;
                let gradient =
                    coupling * *laplacian + radial * radial_delta * phasor + confinement * phasor
                        - stimulus_gain * *field
                        + entropy_gradient(phasor, norm_sqr, entropy_state, entropy_gain);
                let amplitude = norm_sqr.sqrt();
                NodeMetrics {
                    radial: f64::from(radial_delta * radial_delta),
                    confinement: f64::from(norm_sqr),
                    stimulus: f64::from((phasor.conj() * field).re),
                    residual_sqr: f64::from(gradient.norm_sqr()),
                    amplitude: f64::from(amplitude),
                    active: usize::from(amplitude > active_threshold),
                }
            };
        let metrics = if self.node_count() < PARALLEL_NODE_THRESHOLD {
            self.phasors
                .iter()
                .zip(&self.stimulus)
                .zip(&*laplacian)
                .map(contribution)
                .fold(NodeMetrics::default(), NodeMetrics::combine)
        } else {
            self.phasors
                .par_iter()
                .zip(self.stimulus.par_iter())
                .zip(laplacian.par_iter())
                .map(contribution)
                .reduce(NodeMetrics::default, NodeMetrics::combine)
        };

        let radial_energy = 0.25 * radial * metrics.radial as f32;
        let confinement_energy = 0.5 * confinement * metrics.confinement as f32;
        let stimulus_energy = -stimulus_gain * metrics.stimulus as f32;
        let internal_energy =
            coupling_energy + radial_energy + confinement_energy + stimulus_energy;
        let free_energy = internal_energy - entropy_gain * entropy_state.entropy;
        let nodes = self.node_count().max(1) as f64;

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
            gradient_residual: (metrics.residual_sqr / nodes).sqrt() as f32,
            mean_amplitude: (metrics.amplitude / nodes) as f32,
            phase_coherence: self.operator.phase_coherence(&self.phasors),
            active_phasors: metrics.active,
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

    /// Solver canónico de inferencia: minimiza directamente la energía libre
    /// mediante gradiente precondicionado y búsqueda de línea de Armijo. A
    /// diferencia de `step`, no inyecta ruido y sólo acepta movimientos que
    /// reducen F.
    pub fn minimize_free_energy(
        &mut self,
        config: NativePhasorMinimizerConfig,
    ) -> NativePhasorMinimizationReport {
        let config = sanitize_minimizer_config(config);
        let nodes = self.node_count();
        // Buffers persistentes: se extraen del motor (evita conflictos de
        // préstamo con `&self`), se usan y se devuelven al terminar.
        let mut scratch = std::mem::take(&mut self.minimizer_scratch);
        scratch.gradient.clear();
        scratch.gradient.resize(nodes, Complex32::new(0.0, 0.0));
        scratch.direction.clear();
        scratch.direction.resize(nodes, Complex32::new(0.0, 0.0));
        scratch.candidate.clear();
        scratch.candidate.extend_from_slice(&self.phasors);
        scratch.attention.clear();
        scratch.attention.resize(nodes, 1.0);
        scratch.handshake_boundary.clear();
        scratch
            .handshake_boundary
            .resize(nodes, Complex32::new(0.0, 0.0));
        scratch.handshake_buffer.clear();
        scratch
            .handshake_buffer
            .resize(nodes, Complex32::new(0.0, 0.0));
        scratch.handshake_reachability.clear();
        scratch.handshake_reachability.resize(nodes, 0.0);
        let initial = self.report_with_scratch(&mut scratch.gradient);
        let boundary_norm_sqr = self
            .stimulus
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f32>();
        let handshake_applied = config.handshake_strength > 0.0 && boundary_norm_sqr > EPSILON;
        let mut handshake_operator_applications = 0usize;
        let mut reachability_sum = 0.0f64;
        if handshake_applied {
            let seed_norm = boundary_norm_sqr.sqrt().max(EPSILON);
            for (value, seed) in scratch.handshake_boundary.iter_mut().zip(&self.stimulus) {
                *value = *seed / seed_norm;
            }
            for _ in 0..config.handshake_rounds {
                self.operator
                    .propagate_adjoint(&scratch.handshake_boundary, &mut scratch.handshake_buffer);
                for (propagated, seed) in scratch.handshake_buffer.iter_mut().zip(&self.stimulus) {
                    *propagated = *propagated * (1.0 - config.handshake_damping)
                        + (*seed / seed_norm) * config.handshake_damping;
                }
                normalize_complex_field(&mut scratch.handshake_buffer);
                std::mem::swap(
                    &mut scratch.handshake_boundary,
                    &mut scratch.handshake_buffer,
                );
                handshake_operator_applications += 1;
            }
            // La frontera ya no cambia: se separa en dirección unitaria y
            // alcanzabilidad para que cada iteración sólo haga un producto
            // escalar por nodo.
            let peak = scratch
                .handshake_boundary
                .iter()
                .map(|value| value.norm())
                .fold(0.0f32, f32::max)
                .max(EPSILON);
            for (message, reachability) in scratch
                .handshake_boundary
                .iter_mut()
                .zip(&mut scratch.handshake_reachability)
            {
                let magnitude = message.norm();
                *reachability = magnitude / peak;
                *message = if magnitude > EPSILON {
                    *message / magnitude
                } else {
                    Complex32::new(0.0, 0.0)
                };
                reachability_sum += f64::from(*reachability);
            }
            reachability_sum = reachability_sum.max(f64::from(EPSILON));
        }
        let mut current_energy = initial.free_energy;
        let gradient = &mut scratch.gradient;
        let direction = &mut scratch.direction;
        let candidate = &mut scratch.candidate;
        let attention = &mut scratch.attention;
        let handshake_boundary = &scratch.handshake_boundary;
        let handshake_reachability = &scratch.handshake_reachability;
        let mut step_size = config.initial_step;
        let mut iterations = 0;
        let mut energy_evaluations = 0;
        let mut accepted_steps = 0;
        let mut rejected_steps = 0;
        let mut warm_start_applied = false;
        let mut converged = false;
        let mut attention_entropy_sum = 0.0f64;
        let mut attention_updates = 0usize;
        let mut peak_attention_gain = 1.0f32;
        let mut attention_phi_sum = 0.0f64;
        let mut attention_ignitions = 0usize;
        let mut handshake_coherence_sum = 0.0f64;
        let mut handshake_updates = 0usize;
        let adaptive = config.inference_policy == NativePhasorInferencePolicy::Adaptive;
        let mut handshake_saturated = false;
        let mut transaction_agreement = 1.0f32;
        let mut attention_dormant_for = 0usize;
        let mut release_threshold = 0.0f32;
        let mut modifier_release_iteration = usize::MAX;
        let probe_interval = config.attention_probe_interval.max(1);

        if config.topological_warm_start {
            self.topological_phase_warm_start();
            let warm_energy = self.free_energy_for(&self.phasors);
            energy_evaluations += 1;
            if warm_energy <= current_energy {
                current_energy = warm_energy;
                candidate.copy_from_slice(&self.phasors);
                warm_start_applied = true;
            } else {
                self.phasors.copy_from_slice(candidate);
            }
        }

        let sequential = nodes < PARALLEL_NODE_THRESHOLD;
        let coupling = self.config.coupling_strength;
        let radial = self.config.radial_strength;
        let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
        let confinement = self.config.confinement;
        let max_amplitude = self.config.max_amplitude;
        let max_amplitude_sqr = max_amplitude * max_amplitude;

        for iteration in 0..config.max_iterations {
            self.free_energy_gradient(&self.phasors, gradient);
            let residual_sum = if sequential {
                gradient
                    .iter()
                    .map(|value| f64::from(value.norm_sqr()))
                    .sum::<f64>()
            } else {
                gradient
                    .par_iter()
                    .map(|value| f64::from(value.norm_sqr()))
                    .sum::<f64>()
            };
            let residual = (residual_sum / nodes.max(1) as f64).sqrt() as f32;
            if residual <= config.residual_tolerance {
                converged = true;
                break;
            }
            if iteration == 0 {
                release_threshold = residual * config.modifier_release_residual_ratio;
            }
            // Los moduladores reorientan; cerca del mínimo el residuo manda y
            // reorientar sólo desperdicia presupuesto de línea.
            let released = adaptive && residual <= release_threshold;
            if released && modifier_release_iteration == usize::MAX {
                modifier_release_iteration = iteration;
            }
            let use_handshake =
                handshake_applied && nodes > 1 && !released && !(adaptive && handshake_saturated);
            let attention_enabled = config.attention_strength > 0.0 && nodes > 1 && !released;
            let probe_attention = attention_enabled
                && (!adaptive || attention_dormant_for == 0 || iteration % probe_interval == 0);

            // Primero se forma la dirección de Jacobi. La atención no cambia
            // el objetivo: aplica una diagonal positiva calculada con un
            // softmax del residuo por nodo. Armijo sigue validando F exacta.
            let precondition = |gradient: Complex32, phasor: Complex32, diagonal: f32| {
                let denominator = if config.jacobi_preconditioner {
                    (coupling * diagonal + radial * (phasor.norm_sqr() + target_sqr) + confinement)
                        .max(EPSILON)
                } else {
                    1.0
                };
                gradient / denominator
            };
            if sequential {
                direction
                    .iter_mut()
                    .zip(&*gradient)
                    .zip(&self.phasors)
                    .zip(&self.operator.diagonal)
                    .for_each(|(((value, gradient), phasor), diagonal)| {
                        *value = precondition(*gradient, *phasor, *diagonal);
                    })
            } else {
                direction
                    .par_iter_mut()
                    .zip(gradient.par_iter())
                    .zip(self.phasors.par_iter())
                    .zip(self.operator.diagonal.par_iter())
                    .for_each(|(((value, gradient), phasor), diagonal)| {
                        *value = precondition(*gradient, *phasor, *diagonal);
                    })
            }

            // Derivada direccional y saliencia máxima de la dirección Jacobi
            // en un único recorrido: ambas se necesitan antes de cualquier
            // modulador y recorrerlas por separado costaba dos pasadas.
            let mut baseline_derivative = 0.0f64;
            let mut peak_direction_norm = 0.0f32;
            let mut direction_norm_sum = 0.0f64;
            for (value, gradient) in direction.iter().zip(&*gradient) {
                baseline_derivative += f64::from((gradient.conj() * *value).re);
                let norm = value.norm();
                peak_direction_norm = peak_direction_norm.max(norm);
                direction_norm_sum += f64::from(norm);
            }
            let baseline_derivative = baseline_derivative.max(f64::from(EPSILON));
            // Cada modulador se renormaliza al mismo presupuesto direccional,
            // así que el factor acumulado se propaga como escalar y se aplica
            // al proponer: evita una pasada de reescalado por modulador y deja
            // la derivada direccional efectiva invariante frente a Armijo.
            let mut direction_scale = 1.0f32;

            if use_handshake {
                let mut score_sum = 0.0f64;
                let mut coherence_sum = 0.0f64;
                for ((weight, phasor), (boundary, reachability)) in attention
                    .iter_mut()
                    .zip(&self.phasors)
                    .zip(handshake_boundary.iter().zip(handshake_reachability))
                {
                    let amplitude = phasor.norm();
                    let coherence = if amplitude > EPSILON {
                        ((phasor.conj() * *boundary).re / amplitude).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    // La frontera debe focalizar correcciones pendientes, no
                    // amplificar nodos que ya están alineados con ella.
                    *weight = EPSILON + *reachability * (0.5 - 0.5 * coherence);
                    score_sum += f64::from(*weight);
                    coherence_sum += f64::from(coherence * *reachability);
                }
                let mean_score = (score_sum / nodes as f64).max(f64::from(EPSILON));
                let mut guided_derivative = 0.0f64;
                peak_direction_norm = 0.0;
                direction_norm_sum = 0.0;
                for ((value, score), gradient) in
                    direction.iter_mut().zip(&*attention).zip(&*gradient)
                {
                    let relative_score = (f64::from(*score) / mean_score) as f32;
                    let gain = (1.0 + config.handshake_strength * (relative_score - 1.0))
                        .clamp(0.05, config.handshake_max_gain);
                    *value *= gain;
                    guided_derivative += f64::from((gradient.conj() * *value).re);
                    let norm = value.norm();
                    peak_direction_norm = peak_direction_norm.max(norm);
                    direction_norm_sum += f64::from(norm);
                }
                direction_scale =
                    (baseline_derivative / guided_derivative.max(f64::from(EPSILON))) as f32;
                // Coseno medio ponderado por alcanzabilidad: normalizar por el
                // número de nodos diluiría la medida con la dispersión de la
                // frontera y dejaría el umbral de saturación inalcanzable.
                let mean_coherence = (coherence_sum / reachability_sum).clamp(-1.0, 1.0);
                handshake_coherence_sum += mean_coherence;
                handshake_updates += 1;
                handshake_saturated =
                    mean_coherence as f32 >= config.handshake_saturation_coherence;
                // ⟨Φ|Ψ⟩ reescalado a [0,1]: es el grado en que la evidencia
                // hacia adelante y la frontera hacia atrás están en fase.
                transaction_agreement = (0.5 + 0.5 * mean_coherence) as f32;
            }

            // Capa 2: la atención opera después de la intención Handshake. El
            // softmax detecta saliencia en la dirección ya guiada y Φ decide
            // si existe integración de fase suficiente para encender el foco.
            if probe_attention {
                let inverse_temperature = config.attention_temperature.recip();
                // La saliencia se mide contra la media de la propia dirección.
                // Sobre módulos absolutos, `ln(1+x)` con x≪1 comprime hasta
                // borrar el contraste: cerca del mínimo una razón de 10 a 1
                // entre nodos deja centésimas de nat y el softmax queda plano
                // por muy focalizado que esté el error. En términos relativos
                // el foco sólo depende de la forma del residuo, no de su
                // escala, así que la atención no se apaga al converger.
                let mean_direction_norm =
                    (direction_norm_sum / nodes as f64).max(f64::from(EPSILON)) as f32;
                // `ln(1+x)` es monótona, así que el máximo de los logits sale
                // del módulo máximo ya observado sin recorrer la dirección.
                let max_logit =
                    (peak_direction_norm / mean_direction_norm).ln_1p() * inverse_temperature;
                let mut normalizer = 0.0f64;
                let mut weighted_log_weight = 0.0f64;
                let mut phase_resultant = Complex32::new(0.0, 0.0);
                for ((weight, value), phasor) in
                    attention.iter_mut().zip(&*direction).zip(&self.phasors)
                {
                    let logit = (value.norm() / mean_direction_norm).ln_1p() * inverse_temperature
                        - max_logit;
                    *weight = logit.exp();
                    normalizer += f64::from(*weight);
                    weighted_log_weight += f64::from(*weight) * f64::from(logit);
                    let amplitude = phasor.norm();
                    if amplitude > EPSILON {
                        phase_resultant += (*phasor / amplitude) * *weight;
                    }
                }
                let normalizer = normalizer.max(f64::from(EPSILON));
                // Entropía en forma cerrada sobre sumas suficientes: evita el
                // segundo recorrido que calculaba `-Σ p ln p` término a término.
                let entropy = (normalizer.ln() - weighted_log_weight / normalizer).max(0.0);
                let normalized_entropy = (entropy / (nodes as f64).ln()).clamp(0.0, 1.0);
                let resultant = f64::from(phase_resultant.norm()) / normalizer;
                // Φ es la interferencia entre los dos vectores de estado: foco
                // no uniforme, integración global de fase y acuerdo entre la
                // evidencia hacia adelante y la frontera hacia atrás. Sin
                // frontera el acuerdo vale 1 y Φ queda como concentración pura.
                let phi =
                    ((1.0 - normalized_entropy) * resultant * f64::from(transaction_agreement))
                        .clamp(0.0, 1.0);
                attention_phi_sum += phi;
                attention_entropy_sum += normalized_entropy;
                attention_updates += 1;
                if phi as f32 >= config.attention_ignition_threshold {
                    attention_ignitions += 1;
                    attention_dormant_for = 0;
                    let mean_weight = normalizer / nodes as f64;
                    let mut local_peak_gain = 1.0f32;
                    let mut attended_derivative = 0.0f64;
                    for ((value, weight), gradient) in
                        direction.iter_mut().zip(&*attention).zip(&*gradient)
                    {
                        let relative_weight = (f64::from(*weight) / mean_weight) as f32;
                        let gain = (1.0 + config.attention_strength * (relative_weight - 1.0))
                            .clamp(0.05, config.attention_max_gain);
                        *value *= gain;
                        local_peak_gain = local_peak_gain.max(gain);
                        attended_derivative += f64::from((gradient.conj() * *value).re);
                    }
                    direction_scale =
                        (baseline_derivative / attended_derivative.max(f64::from(EPSILON))) as f32;
                    peak_attention_gain =
                        peak_attention_gain.max(local_peak_gain * direction_scale);
                } else {
                    attention_dormant_for += 1;
                }
            }

            // Invariante del diseño: todo modulador se renormaliza al mismo
            // presupuesto, así que la derivada efectiva coincide siempre con la
            // de Jacobi y no hace falta recalcularla.
            let directional_derivative = baseline_derivative as f32;

            let mut trial_step = step_size;
            let mut accepted_energy = current_energy;
            let mut accepted = false;
            for _ in 0..config.max_backtracks {
                let effective_step = trial_step * direction_scale;
                let propose = |phasor: Complex32, direction: Complex32| {
                    let mut next = phasor - direction * effective_step;
                    let amplitude_sqr = next.norm_sqr();
                    if amplitude_sqr > max_amplitude_sqr {
                        next *= max_amplitude / amplitude_sqr.sqrt();
                    }
                    next
                };
                if sequential {
                    candidate
                        .iter_mut()
                        .zip(&self.phasors)
                        .zip(&*direction)
                        .for_each(|((value, phasor), direction)| {
                            *value = propose(*phasor, *direction);
                        });
                } else {
                    candidate
                        .par_iter_mut()
                        .zip(self.phasors.par_iter())
                        .zip(direction.par_iter())
                        .for_each(|((value, phasor), direction)| {
                            *value = propose(*phasor, *direction);
                        });
                }
                let trial_energy = self.free_energy_for(candidate);
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
            self.phasors.copy_from_slice(candidate);
            self.tick = self.tick.wrapping_add(1);
            current_energy = accepted_energy;
            accepted_steps += 1;
            step_size = (trial_step * config.step_growth).min(config.max_step);
            if energy_delta <= config.energy_tolerance && residual <= config.residual_tolerance {
                converged = true;
                break;
            }
        }

        let final_report = self.report_with_scratch(&mut scratch.gradient);
        if final_report.gradient_residual <= config.residual_tolerance {
            converged = true;
        }
        self.minimizer_scratch = scratch;
        NativePhasorMinimizationReport {
            initial,
            final_report,
            iterations,
            energy_evaluations,
            accepted_steps,
            rejected_steps,
            warm_start_applied,
            converged,
            mean_attention_entropy: if attention_updates == 0 {
                1.0
            } else {
                (attention_entropy_sum / attention_updates as f64) as f32
            },
            peak_attention_gain,
            mean_integrated_information: if attention_updates == 0 {
                0.0
            } else {
                (attention_phi_sum / attention_updates as f64) as f32
            },
            attention_ignitions,
            handshake_applied,
            handshake_operator_applications,
            mean_handshake_coherence: if handshake_updates == 0 {
                0.0
            } else {
                (handshake_coherence_sum / handshake_updates as f64) as f32
            },
            handshake_iterations: handshake_updates,
            attention_probes: attention_updates,
            modifier_release_iteration: modifier_release_iteration.min(iterations),
        }
    }

    /// Energía libre del estado actual sin las métricas derivadas de
    /// [`Self::report`]. Ahorra el producto Laplaciano y el recorrido de
    /// coherencia cuando sólo se necesita F.
    pub fn free_energy(&self) -> f32 {
        self.free_energy_for(&self.phasors)
    }

    /// Ejecuta la variante experimental de inferencia activa local mediante
    /// Metropolis-within-Gibbs y actualizaciones coordenadas de energía libre.
    ///
    /// La aceptación Gibbs usa deltas exactos calculados sobre el vecindario
    /// del nodo. La entropía se mantiene con estadísticos suficientes O(1);
    /// `sampled_entropy` es una estimación Monte Carlo independiente de ese
    /// valor y permite medir el error introducido por muestreo.
    pub fn active_inference(
        &mut self,
        config: NativePhasorActiveInferenceConfig,
    ) -> NativePhasorActiveInferenceReport {
        let config = sanitize_active_inference_config(config);
        let initial = self.report();
        let mut entropy = entropy_state(&self.phasors);
        let mut current_free_energy = initial.free_energy;
        let mut best_free_energy = current_free_energy;
        let mut best_state = self.phasors.clone();
        let mut gibbs_proposals = 0usize;
        let mut gibbs_accepted = 0usize;
        let mut local_updates_accepted = 0usize;
        let nodes = self.node_count();

        for sweep in 0..config.sweeps {
            let start = (splitmix64(config.seed ^ sweep as u64) as usize) % nodes;
            for offset in 0..nodes {
                let node = (start + offset) % nodes;
                let counter = (sweep as u64)
                    .wrapping_mul(nodes as u64)
                    .wrapping_add(offset as u64);
                let current = self.phasors[node];
                let proposal = current
                    + Complex32::new(
                        gaussian_from_counter(config.seed, counter.wrapping_mul(2)),
                        gaussian_from_counter(
                            config.seed.rotate_left(23),
                            counter.wrapping_mul(2).wrapping_add(1),
                        ),
                    ) * config.proposal_std;
                gibbs_proposals += 1;
                if proposal.norm() <= self.config.max_amplitude {
                    let (delta_free_energy, next_entropy) =
                        self.local_free_energy_delta(node, proposal, entropy);
                    let accept_probability = if delta_free_energy <= 0.0 {
                        1.0
                    } else {
                        (-delta_free_energy / config.sampling_temperature).exp()
                    };
                    let draw = unit_from_u64(splitmix64(
                        config.seed.rotate_left(41) ^ counter ^ 0x91E1_0DA5_C79E_7B1D,
                    ));
                    if draw < accept_probability {
                        self.phasors[node] = proposal;
                        entropy = next_entropy;
                        current_free_energy += delta_free_energy;
                        gibbs_accepted += 1;
                    }
                }

                if config.local_learning_rate > 0.0 {
                    let current = self.phasors[node];
                    let gradient = self.local_free_energy_gradient(node, entropy);
                    let denominator = (self.config.coupling_strength
                        * self.operator.diagonal[node]
                        + self.config.radial_strength
                            * (current.norm_sqr()
                                + self.config.target_amplitude * self.config.target_amplitude)
                        + self.config.confinement)
                        .max(EPSILON);
                    let candidate = current - gradient * (config.local_learning_rate / denominator);
                    if candidate.norm() <= self.config.max_amplitude {
                        let (delta_free_energy, next_entropy) =
                            self.local_free_energy_delta(node, candidate, entropy);
                        if delta_free_energy <= 0.0 {
                            self.phasors[node] = candidate;
                            entropy = next_entropy;
                            current_free_energy += delta_free_energy;
                            local_updates_accepted += 1;
                        }
                    }
                }
            }
            self.tick = self.tick.wrapping_add(1);
            if sweep + 1 >= config.burn_in_sweeps && current_free_energy < best_free_energy {
                best_free_energy = current_free_energy;
                best_state.copy_from_slice(&self.phasors);
            }
        }

        if config.keep_best && best_free_energy < current_free_energy {
            self.phasors.copy_from_slice(&best_state);
        }
        let final_report = self.report();
        let (sampled_mean_internal_energy, sampled_entropy) =
            self.sample_observables(config.entropy_samples, config.seed.rotate_left(7));
        NativePhasorActiveInferenceReport {
            initial,
            final_report,
            best_free_energy: best_free_energy.min(final_report.free_energy),
            sweeps: config.sweeps,
            gibbs_proposals,
            gibbs_accepted,
            local_updates_accepted,
            sampled_mean_internal_energy,
            sampled_entropy,
            entropy_absolute_error: (sampled_entropy - final_report.entropy).abs(),
        }
    }

    /// Mínimo global analítico para el caso no frustrado, sin estímulo y a
    /// temperatura efectiva cero. Sirve como oráculo de correctitud.
    pub fn analytic_unfrustrated_minimum(&self) -> Option<f32> {
        if self
            .operator
            .edges
            .iter()
            // `transport` es `exp(-i*phase)`, así que `re`/`im` ya contienen
            // `cos(phase)` y `-sin(phase)` sin recalcular la trigonometría.
            .any(|edge| edge.transport.im.abs() > 1.0e-6 || 1.0 - edge.transport.re > 1.0e-6)
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
        let entropy_gain = self.entropy_gain();
        let entropy_state = if entropy_gain != 0.0 {
            entropy_state(phasors)
        } else {
            EntropyState::default()
        };
        let coupling = self.config.coupling_strength;
        let radial = self.config.radial_strength;
        let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
        let confinement = self.config.confinement;
        let stimulus_gain = self.config.stimulus_gain;
        let complete_gradient = |laplacian: Complex32, phasor: Complex32, field: Complex32| {
            let norm_sqr = phasor.norm_sqr();
            coupling * laplacian + radial * (norm_sqr - target_sqr) * phasor + confinement * phasor
                - stimulus_gain * field
                + entropy_gradient(phasor, norm_sqr, entropy_state, entropy_gain)
        };
        if output.len() < PARALLEL_NODE_THRESHOLD {
            output.iter_mut().zip(phasors).zip(&self.stimulus).for_each(
                |((value, phasor), field)| {
                    *value = complete_gradient(*value, *phasor, *field);
                },
            );
        } else {
            output
                .par_iter_mut()
                .zip(phasors.par_iter())
                .zip(self.stimulus.par_iter())
                .for_each(|((value, phasor), field)| {
                    *value = complete_gradient(*value, *phasor, *field);
                });
        }
    }

    fn local_free_energy_gradient(&self, node: usize, entropy: EntropyState) -> Complex32 {
        let current = self.phasors[node];
        let mut laplacian = current * self.operator.diagonal[node];
        for cursor in self.operator.row_offsets[node]..self.operator.row_offsets[node + 1] {
            laplacian -= self.phasors[self.operator.columns[cursor]]
                * self.operator.transports[cursor]
                * self.operator.weights[cursor];
        }
        self.config.coupling_strength * laplacian
            + self.config.radial_strength
                * (current.norm_sqr() - self.config.target_amplitude * self.config.target_amplitude)
                * current
            + self.config.confinement * current
            - self.config.stimulus_gain * self.stimulus[node]
            + entropy_gradient(
                current,
                current.norm_sqr(),
                entropy,
                self.config.entropy_weight * self.effective_temperature(),
            )
    }

    fn local_free_energy_delta(
        &self,
        node: usize,
        candidate: Complex32,
        entropy: EntropyState,
    ) -> (f32, EntropyState) {
        let current = self.phasors[node];
        let mut coupling_delta = 0.0;
        for cursor in self.operator.row_offsets[node]..self.operator.row_offsets[node + 1] {
            let neighbor = self.phasors[self.operator.columns[cursor]];
            let transport = self.operator.transports[cursor];
            let weight = self.operator.weights[cursor];
            coupling_delta += 0.5
                * self.config.coupling_strength
                * weight
                * ((candidate - transport * neighbor).norm_sqr()
                    - (current - transport * neighbor).norm_sqr());
        }
        let onsite = |value: Complex32| {
            let radial_delta =
                value.norm_sqr() - self.config.target_amplitude * self.config.target_amplitude;
            0.25 * self.config.radial_strength * radial_delta * radial_delta
                + 0.5 * self.config.confinement * value.norm_sqr()
                - self.config.stimulus_gain * (value.conj() * self.stimulus[node]).re
        };
        let current_q = current.norm_sqr() + EPSILON;
        let candidate_q = candidate.norm_sqr() + EPSILON;
        let next_entropy = entropy_from_sums(
            entropy.q_sum + candidate_q - current_q,
            entropy.q_log_q_sum + candidate_q * candidate_q.ln() - current_q * current_q.ln(),
        );
        let entropy_delta = next_entropy.entropy - entropy.entropy;
        (
            coupling_delta + onsite(candidate)
                - onsite(current)
                - self.config.entropy_weight * self.effective_temperature() * entropy_delta,
            next_entropy,
        )
    }

    fn sample_observables(&self, samples: usize, seed: u64) -> (f32, f32) {
        let samples = samples.max(1);
        let q_sum = self
            .phasors
            .iter()
            .map(|phasor| phasor.norm_sqr() + EPSILON)
            .sum::<f32>()
            .max(EPSILON);
        let mut cumulative = Vec::with_capacity(self.node_count());
        let mut running = 0.0;
        for phasor in &self.phasors {
            running += (phasor.norm_sqr() + EPSILON) / q_sum;
            cumulative.push(running);
        }
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }

        let mut entropy_sum = 0.0;
        let mut energy_sum = 0.0;
        for sample in 0..samples {
            let entropy_draw = unit_from_u64(splitmix64(seed ^ sample as u64));
            let sampled_node = cumulative.partition_point(|value| *value < entropy_draw);
            let probability = (self.phasors[sampled_node].norm_sqr() + EPSILON) / q_sum;
            entropy_sum -= probability.max(EPSILON).ln();

            let energy_node =
                (splitmix64(seed.rotate_left(31) ^ sample as u64 ^ 0xA24B_AED4_963E_E407) as usize)
                    % self.node_count();
            energy_sum += self.node_internal_energy_share(energy_node);
        }
        (
            energy_sum * self.node_count() as f32 / samples as f32,
            entropy_sum / samples as f32,
        )
    }

    fn node_internal_energy_share(&self, node: usize) -> f32 {
        let current = self.phasors[node];
        let mut coupling_share = 0.0;
        for cursor in self.operator.row_offsets[node]..self.operator.row_offsets[node + 1] {
            let neighbor = self.phasors[self.operator.columns[cursor]];
            coupling_share += 0.25
                * self.config.coupling_strength
                * self.operator.weights[cursor]
                * (current - self.operator.transports[cursor] * neighbor).norm_sqr();
        }
        let radial_delta =
            current.norm_sqr() - self.config.target_amplitude * self.config.target_amplitude;
        coupling_share
            + 0.25 * self.config.radial_strength * radial_delta * radial_delta
            + 0.5 * self.config.confinement * current.norm_sqr()
            - self.config.stimulus_gain * (current.conj() * self.stimulus[node]).re
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
            let root_norm = self.phasors[root].norm_sqr().sqrt();
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
        let resynchronize = |phasor: &mut Complex32, synchronized: &Complex32| {
            let amplitude = if can_use_radial_equilibrium {
                equilibrium_amplitude
            } else {
                phasor.norm_sqr().sqrt()
            };
            *phasor = synchronized * amplitude;
        };
        if nodes < PARALLEL_NODE_THRESHOLD {
            self.phasors
                .iter_mut()
                .zip(&synchronized)
                .for_each(|(phasor, target)| {
                    resynchronize(phasor, target);
                });
        } else {
            self.phasors
                .par_iter_mut()
                .zip(synchronized.par_iter())
                .for_each(|(phasor, target)| resynchronize(phasor, target));
        }
    }

    /// Evaluación completa de F para un estado candidato. Es la operación que
    /// más veces se repite por iteración (una por retroceso de Armijo), así que
    /// los términos local y entrópico comparten un único recorrido.
    fn free_energy_for(&self, phasors: &[Complex32]) -> f32 {
        let coupling = self
            .operator
            .coupling_energy(phasors, self.config.coupling_strength);
        let target_sqr = self.config.target_amplitude * self.config.target_amplitude;
        let radial = self.config.radial_strength;
        let confinement = self.config.confinement;
        let stimulus_gain = self.config.stimulus_gain;
        let entropy_gain = self.entropy_gain();
        let track_entropy = entropy_gain != 0.0;
        let contribution = |(phasor, field): (&Complex32, &Complex32)| {
            let norm_sqr = phasor.norm_sqr();
            let radial_delta = norm_sqr - target_sqr;
            let local = 0.25 * radial * radial_delta * radial_delta + 0.5 * confinement * norm_sqr
                - stimulus_gain * (phasor.conj() * field).re;
            if track_entropy {
                let q = norm_sqr + EPSILON;
                (f64::from(local), f64::from(q), f64::from(q * q.ln()))
            } else {
                (f64::from(local), 0.0, 0.0)
            }
        };
        let combine = |left: (f64, f64, f64), right: (f64, f64, f64)| {
            (left.0 + right.0, left.1 + right.1, left.2 + right.2)
        };
        let (local, q_sum, q_log_q_sum) = if phasors.len() < PARALLEL_NODE_THRESHOLD {
            phasors
                .iter()
                .zip(&self.stimulus)
                .map(contribution)
                .fold((0.0, 0.0, 0.0), combine)
        } else {
            phasors
                .par_iter()
                .zip(self.stimulus.par_iter())
                .map(contribution)
                .reduce(|| (0.0, 0.0, 0.0), combine)
        };
        let local = local as f32;
        if !track_entropy {
            return coupling + local;
        }
        coupling + local - entropy_gain * entropy_from_wide_sums(q_sum, q_log_q_sum).entropy
    }

    #[inline]
    fn effective_temperature(&self) -> f32 {
        self.mean_temperature.max(0.0) * self.config.temperature_scale
    }

    /// Peso `entropy_weight * T_efectiva` del término `-T S`. Cuando vale
    /// exactamente cero el término entero se anula y el recorrido O(N) con un
    /// `ln` por nodo que calcula la entropía puede omitirse sin alterar el
    /// resultado en punto flotante.
    #[inline]
    fn entropy_gain(&self) -> f32 {
        self.config.entropy_weight * self.effective_temperature()
    }
}

fn normalize_complex_field(values: &mut [Complex32]) {
    let norm = values
        .iter()
        .map(|value| f64::from(value.norm_sqr()))
        .sum::<f64>()
        .sqrt() as f32;
    if norm > EPSILON {
        values.iter_mut().for_each(|value| *value /= norm);
    }
}

fn mean_temperature(temperature: &[f32]) -> f32 {
    temperature.iter().copied().sum::<f32>() / temperature.len().max(1) as f32
}

/// Acumulador de las métricas por nodo que publica `report`, agrupadas para
/// que un único recorrido del campo fasorial las produzca todas. Suma en `f64`
/// por la misma razón que [`entropy_from_wide_sums`].
#[derive(Clone, Copy, Debug, Default)]
struct NodeMetrics {
    radial: f64,
    confinement: f64,
    stimulus: f64,
    residual_sqr: f64,
    amplitude: f64,
    active: usize,
}

impl NodeMetrics {
    #[inline]
    fn combine(left: Self, right: Self) -> Self {
        Self {
            radial: left.radial + right.radial,
            confinement: left.confinement + right.confinement,
            stimulus: left.stimulus + right.stimulus,
            residual_sqr: left.residual_sqr + right.residual_sqr,
            amplitude: left.amplitude + right.amplitude,
            active: left.active + right.active,
        }
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
        (f64::from(q), f64::from(q * q.ln()))
    };
    let combine = |left: (f64, f64), right: (f64, f64)| (left.0 + right.0, left.1 + right.1);
    let (q_sum, q_log_q_sum) = if phasors.len() < PARALLEL_NODE_THRESHOLD {
        phasors.iter().map(contribution).fold((0.0, 0.0), combine)
    } else {
        phasors
            .par_iter()
            .map(contribution)
            .reduce(|| (0.0, 0.0), combine)
    };
    entropy_from_wide_sums(q_sum, q_log_q_sum)
}

/// Cierre de la entropía a partir de acumuladores de doble precisión.
///
/// Los recorridos por debajo del umbral de paralelismo suman en serie, y una
/// suma encadenada en `f32` sobre decenas de miles de nodos arrastra un error
/// proporcional a `n`. Acumular en `f64` mantiene la energía libre tan precisa
/// como la reducción en árbol de `rayon` a cualquier tamaño.
fn entropy_from_wide_sums(q_sum: f64, q_log_q_sum: f64) -> EntropyState {
    if q_sum <= f64::from(EPSILON) {
        EntropyState::default()
    } else {
        EntropyState {
            entropy: (q_sum.ln() - q_log_q_sum / q_sum).max(0.0) as f32,
            q_sum: q_sum as f32,
            q_log_q_sum: q_log_q_sum as f32,
        }
    }
}

fn entropy_from_sums(q_sum: f32, q_log_q_sum: f32) -> EntropyState {
    if q_sum <= EPSILON {
        EntropyState::default()
    } else {
        EntropyState {
            entropy: (q_sum.ln() - q_log_q_sum / q_sum).max(0.0),
            q_sum,
            q_log_q_sum,
        }
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
        attention_strength: config.attention_strength.clamp(0.0, 1.0),
        attention_temperature: config.attention_temperature.max(EPSILON),
        attention_max_gain: config.attention_max_gain.max(1.0),
        attention_ignition_threshold: config.attention_ignition_threshold.clamp(0.0, 1.0),
        handshake_strength: config.handshake_strength.clamp(0.0, 1.0),
        handshake_rounds: config.handshake_rounds.max(1),
        handshake_damping: config.handshake_damping.clamp(0.0, 1.0),
        handshake_max_gain: config.handshake_max_gain.max(1.0),
        modifier_release_residual_ratio: config.modifier_release_residual_ratio.clamp(0.0, 1.0),
        handshake_saturation_coherence: config.handshake_saturation_coherence.clamp(-1.0, 1.0),
        attention_probe_interval: config.attention_probe_interval.max(1),
        ..config
    }
}

fn sanitize_active_inference_config(
    config: NativePhasorActiveInferenceConfig,
) -> NativePhasorActiveInferenceConfig {
    NativePhasorActiveInferenceConfig {
        sweeps: config.sweeps.max(1),
        burn_in_sweeps: config.burn_in_sweeps.min(config.sweeps.max(1)),
        sampling_temperature: config.sampling_temperature.max(EPSILON),
        proposal_std: config.proposal_std.max(EPSILON),
        local_learning_rate: config.local_learning_rate.clamp(0.0, 1.0),
        entropy_samples: config.entropy_samples.max(1),
        ..config
    }
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
    fn local_active_inference_reduces_free_energy_without_global_evaluations() {
        let core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: 32,
            spatial_degree: 2,
            temporal_degree: 1,
            temperature: 0.5,
            seed: 72_026,
            ..NativeThermoCdtConfig::default()
        });
        let mut engine = NativePhasorThermodynamicEngine::from_core(
            &core,
            NativePhasorConfig {
                radial_strength: 4.0,
                entropy_weight: 0.15,
                temperature_scale: 0.1,
                noise_scale: 0.0,
                ..NativePhasorConfig::default()
            },
        )
        .unwrap();
        for (node, phasor) in engine.phasors.iter_mut().enumerate() {
            *phasor = Complex32::from_polar(0.8 + 0.01 * node as f32, node as f32 * 0.71);
        }
        let before = engine.report();
        let result = engine.active_inference(NativePhasorActiveInferenceConfig {
            sweeps: 40,
            burn_in_sweeps: 0,
            sampling_temperature: 1.0e-4,
            local_learning_rate: 0.35,
            entropy_samples: 256,
            ..NativePhasorActiveInferenceConfig::default()
        });

        assert!(
            result.final_report.free_energy < before.free_energy,
            "{result:?}"
        );
        assert!(result.local_updates_accepted > 0, "{result:?}");
        assert_eq!(result.gibbs_proposals, 40 * engine.node_count());
    }

    #[test]
    fn zero_handshake_strength_is_bit_exact_armijo() {
        let core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: 32,
            temperature: 0.0,
            seed: 81_026,
            ..NativeThermoCdtConfig::default()
        });
        let mut baseline =
            NativePhasorThermodynamicEngine::from_core(&core, NativePhasorConfig::default())
                .unwrap();
        baseline.inject_pattern(&[3, 7, 11], 1.0, 0.4);
        let mut control = baseline.clone();
        let baseline_report = baseline.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: 20,
            topological_warm_start: false,
            ..NativePhasorMinimizerConfig::default()
        });
        let control_report = control.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: 20,
            topological_warm_start: false,
            handshake_strength: 0.0,
            handshake_rounds: 9,
            ..NativePhasorMinimizerConfig::default()
        });

        assert_eq!(baseline.phasors, control.phasors);
        assert_eq!(baseline_report, control_report);
        assert!(!control_report.handshake_applied);
        assert_eq!(control_report.handshake_operator_applications, 0);
    }

    #[test]
    fn handshake_uses_the_existing_stimulus_and_preserves_armijo_descent() {
        let core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: 48,
            temperature: 0.0,
            seed: 82_026,
            ..NativeThermoCdtConfig::default()
        });
        let mut engine =
            NativePhasorThermodynamicEngine::from_core(&core, NativePhasorConfig::default())
                .unwrap();
        engine.inject_pattern(&[2, 13, 29], 1.0, 0.7);
        let report = engine.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: 24,
            topological_warm_start: false,
            handshake_strength: 0.65,
            handshake_rounds: 3,
            ..NativePhasorMinimizerConfig::default()
        });

        assert!(report.handshake_applied, "{report:?}");
        assert_eq!(report.handshake_operator_applications, 3);
        assert!(
            report.final_report.free_energy <= report.initial.free_energy + 1.0e-6,
            "{report:?}"
        );
    }

    #[test]
    fn attention_phi_ignites_after_handshake_when_threshold_is_reached() {
        let core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: 32,
            temperature: 0.0,
            seed: 83_026,
            ..NativeThermoCdtConfig::default()
        });
        let mut engine =
            NativePhasorThermodynamicEngine::from_core(&core, NativePhasorConfig::default())
                .unwrap();
        engine.inject_pattern(&[1, 5, 9], 1.0, 0.3);
        let report = engine.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: 12,
            residual_tolerance: 0.0,
            energy_tolerance: 0.0,
            topological_warm_start: false,
            handshake_strength: 0.5,
            attention_strength: 0.5,
            attention_ignition_threshold: 0.0,
            inference_policy: NativePhasorInferencePolicy::Fixed,
            ..NativePhasorMinimizerConfig::default()
        });

        assert!(report.handshake_applied, "{report:?}");
        assert_eq!(report.attention_ignitions, report.iterations);
        assert!(report.mean_integrated_information >= 0.0);
        assert!(
            report.final_report.free_energy <= report.initial.free_energy + 1.0e-6,
            "{report:?}"
        );
    }

    fn modulated_engine(seed: u64) -> NativePhasorThermodynamicEngine {
        let core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: 96,
            spatial_degree: 3,
            temperature: 0.0,
            seed,
            ..NativeThermoCdtConfig::default()
        });
        let mut engine =
            NativePhasorThermodynamicEngine::from_core(&core, NativePhasorConfig::default())
                .unwrap();
        for (node, phasor) in engine.phasors.iter_mut().enumerate() {
            *phasor = Complex32::from_polar(0.8 + 0.004 * node as f32, node as f32 * 0.83);
        }
        engine.inject_pattern(&[4, 17, 41, 68], 1.0, 0.35);
        engine
    }

    fn modulated_config(policy: NativePhasorInferencePolicy) -> NativePhasorMinimizerConfig {
        NativePhasorMinimizerConfig {
            max_iterations: 40,
            energy_tolerance: 0.0,
            residual_tolerance: 0.0,
            topological_warm_start: false,
            handshake_strength: 0.65,
            attention_strength: 0.55,
            attention_ignition_threshold: 0.001,
            inference_policy: policy,
            ..NativePhasorMinimizerConfig::default()
        }
    }

    #[test]
    fn attention_keeps_its_focus_while_the_descent_shrinks_the_gradient() {
        // Medida sobre módulos absolutos, la saliencia se apaga al converger:
        // `ln(1+x)` con x≪1 comprime hasta dejar el softmax plano y la
        // atención se queda ciega justo cuando debería ser más selectiva.
        // Contra la media, el foco sólo depende de la forma del residuo.
        let mut engine = modulated_engine(91_026);
        let far = engine.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: 2,
            ..modulated_config(NativePhasorInferencePolicy::Fixed)
        });
        let near = engine.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: 60,
            ..modulated_config(NativePhasorInferencePolicy::Fixed)
        });

        assert!(
            near.final_report.gradient_residual < far.final_report.gradient_residual,
            "{near:?}"
        );
        assert!(near.mean_attention_entropy < 0.995, "{near:?}");
        assert!(near.mean_integrated_information > 1.0e-3, "{near:?}");
    }

    #[test]
    fn adaptive_policy_spends_less_modulation_than_fixed_on_the_same_descent() {
        let mut fixed = modulated_engine(84_026);
        let mut adaptive = modulated_engine(84_026);
        let fixed_report =
            fixed.minimize_free_energy(modulated_config(NativePhasorInferencePolicy::Fixed));
        let adaptive_report =
            adaptive.minimize_free_energy(modulated_config(NativePhasorInferencePolicy::Adaptive));

        assert_eq!(
            fixed_report.modifier_release_iteration,
            fixed_report.iterations
        );
        assert!(
            adaptive_report.handshake_iterations + adaptive_report.attention_probes
                < fixed_report.handshake_iterations + fixed_report.attention_probes,
            "adaptativo={adaptive_report:?} fijo={fixed_report:?}"
        );
        assert!(
            adaptive_report.final_report.free_energy
                <= adaptive_report.initial.free_energy + 1.0e-6,
            "{adaptive_report:?}"
        );
    }

    #[test]
    fn released_modifiers_leave_a_pure_armijo_tail() {
        let mut engine = modulated_engine(85_026);
        let report = engine.minimize_free_energy(NativePhasorMinimizerConfig {
            modifier_release_residual_ratio: 1.0,
            ..modulated_config(NativePhasorInferencePolicy::Adaptive)
        });

        // Con ratio 1.0 el primer residuo ya cumple el umbral de liberación.
        assert_eq!(report.modifier_release_iteration, 0, "{report:?}");
        assert_eq!(report.handshake_iterations, 0, "{report:?}");
        assert_eq!(report.attention_probes, 0, "{report:?}");
    }

    #[test]
    fn modulators_preserve_the_armijo_directional_budget() {
        for policy in [
            NativePhasorInferencePolicy::Fixed,
            NativePhasorInferencePolicy::Adaptive,
        ] {
            let mut engine = modulated_engine(86_026);
            let report = engine.minimize_free_energy(modulated_config(policy));
            // La renormalización deja la derivada direccional invariante, así
            // que ningún modulador puede comprar pasos fuera de presupuesto.
            assert!(
                report.final_report.free_energy <= report.initial.free_energy + 1.0e-6,
                "{policy:?} {report:?}"
            );
            assert!(
                report.energy_evaluations <= report.iterations * 12 + 1,
                "{policy:?} {report:?}"
            );
        }
    }

    #[test]
    fn monte_carlo_entropy_is_exact_for_uniform_amplitudes() {
        let core = two_node_core(0.0);
        let engine =
            NativePhasorThermodynamicEngine::from_core(&core, NativePhasorConfig::default())
                .unwrap();
        let (_, sampled_entropy) = engine.sample_observables(64, 91);
        assert!((sampled_entropy - 2.0_f32.ln()).abs() < 1.0e-6);
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
