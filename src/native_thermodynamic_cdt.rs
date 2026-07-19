use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

const EPSILON: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeCdtEdgeKind {
    Spatial,
    Temporal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeSamplerKind {
    Gibbs,
    Bernoulli,
    Gaussian,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeSamplingConfig {
    pub block_size: usize,
    pub schedule_rounds: usize,
    pub max_blocks_per_pulse: usize,
}

impl Default for NativeSamplingConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            schedule_rounds: 2,
            max_blocks_per_pulse: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeSamplingBlock {
    pub id: usize,
    pub nodes: Vec<usize>,
    pub sampler: NativeSamplerKind,
    pub temperature_scale: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeBlockObservables {
    pub block_id: usize,
    pub nodes: usize,
    pub mean_state: f32,
    pub state_variance: f32,
    pub mean_energy: f32,
    pub mean_activation: f32,
    pub uncertainty: f32,
}

#[derive(Clone, Debug)]
pub struct NativeSamplingProgram {
    pub blocks: Vec<NativeSamplingBlock>,
    pub schedule: Vec<usize>,
    node_to_block: Vec<usize>,
    schedule_rank: Vec<usize>,
    pub config: NativeSamplingConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeThermoCdtConfig {
    pub slices: usize,
    pub nodes_per_slice: usize,
    pub spatial_degree: usize,
    pub temporal_degree: usize,
    pub temperature: f32,
    pub dt: f32,
    pub diffusion: f32,
    pub confinement: f32,
    pub pilot_gain: f32,
    pub phase_coupling: f32,
    pub amplitude_decay: f32,
    pub state_clamp: f32,
    pub seed: u64,
}

impl Default for NativeThermoCdtConfig {
    fn default() -> Self {
        Self {
            slices: 4,
            nodes_per_slice: 160,
            spatial_degree: 4,
            temporal_degree: 2,
            temperature: 0.35,
            dt: 0.012,
            diffusion: 0.18,
            confinement: 0.045,
            pilot_gain: 0.42,
            phase_coupling: 0.16,
            amplitude_decay: 0.002,
            state_clamp: 4.0,
            seed: 0xCD7A_71C0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeThermoCdtReport {
    pub tick: u64,
    pub nodes: usize,
    pub edges: usize,
    pub mean_state: f32,
    pub state_variance: f32,
    pub mean_amplitude: f32,
    pub mean_energy: f32,
    pub free_energy_proxy: f32,
    pub active_nodes: usize,
    pub max_abs_state: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct ThermalStats {
    count: usize,
    state_sum: f64,
    state_sq_sum: f64,
    amplitude_sum: f64,
    energy_sum: f64,
    partition_sum: f64,
    active_nodes: usize,
    max_abs_state: f32,
}

impl ThermalStats {
    #[inline(always)]
    fn merge(self, other: Self) -> Self {
        Self {
            count: self.count + other.count,
            state_sum: self.state_sum + other.state_sum,
            state_sq_sum: self.state_sq_sum + other.state_sq_sum,
            amplitude_sum: self.amplitude_sum + other.amplitude_sum,
            energy_sum: self.energy_sum + other.energy_sum,
            partition_sum: self.partition_sum + other.partition_sum,
            active_nodes: self.active_nodes + other.active_nodes,
            max_abs_state: self.max_abs_state.max(other.max_abs_state),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeThermoCdtSubstrate {
    pub config: NativeThermoCdtConfig,
    pub thermal_state: Vec<f32>,
    pub amplitude: Vec<f32>,
    pub phase: Vec<f32>,
    pub temperature: Vec<f32>,
    pub pilot_force: Vec<f32>,
    pub energy: Vec<f32>,
    pub activation: Vec<f32>,
    pub edge_a: Vec<usize>,
    pub edge_b: Vec<usize>,
    pub edge_kind: Vec<NativeCdtEdgeKind>,
    pub edge_weight: Vec<f32>,
    pub edge_phase: Vec<f32>,
    pub edge_stability: Vec<f32>,
    adjacency_offsets: Vec<usize>,
    adjacency_edges: Vec<usize>,
    previous_state: Vec<f32>,
    previous_phase: Vec<f32>,
    neighborhood_marks: Vec<u32>,
    neighborhood_generation: u32,
    tick: u64,
    rng: Xoshiro256PlusPlus,
}

impl NativeThermoCdtSubstrate {
    pub fn new(config: NativeThermoCdtConfig) -> Self {
        let config = NativeThermoCdtConfig {
            slices: config.slices.max(1),
            nodes_per_slice: config.nodes_per_slice.max(1),
            spatial_degree: config.spatial_degree.max(1),
            temporal_degree: config.temporal_degree.max(1),
            state_clamp: config.state_clamp.abs().max(EPSILON),
            ..config
        };
        let nodes = config.slices * config.nodes_per_slice;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
        let mut thermal_state = Vec::with_capacity(nodes);
        let mut amplitude = Vec::with_capacity(nodes);
        let mut phase = Vec::with_capacity(nodes);

        for idx in 0..nodes {
            thermal_state.push(rng.gen_range(-0.05..0.05));
            amplitude.push(0.5 + 0.5 * rng.gen::<f32>());
            phase.push((idx as f32 * 0.618_034 + rng.gen::<f32>()) % std::f32::consts::TAU);
        }

        let mut substrate = Self {
            config,
            thermal_state,
            amplitude,
            phase,
            temperature: vec![config.temperature.max(0.0); nodes],
            pilot_force: vec![0.0; nodes],
            energy: vec![0.0; nodes],
            activation: vec![0.0; nodes],
            edge_a: Vec::new(),
            edge_b: Vec::new(),
            edge_kind: Vec::new(),
            edge_weight: Vec::new(),
            edge_phase: Vec::new(),
            edge_stability: Vec::new(),
            adjacency_offsets: Vec::new(),
            adjacency_edges: Vec::new(),
            previous_state: vec![0.0; nodes],
            previous_phase: vec![0.0; nodes],
            neighborhood_marks: vec![0; nodes],
            neighborhood_generation: 0,
            tick: 0,
            rng,
        };
        substrate.build_foliated_cdt_graph();
        substrate
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn node_count(&self) -> usize {
        self.thermal_state.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edge_a.len()
    }

    /// Añade slices completos conservando los estados de todos los nodos existentes.
    /// La topología se recompila para incorporar los nodos nuevos.
    pub fn grow_slices(&mut self, additional_slices: usize) -> usize {
        if additional_slices == 0 {
            return 0;
        }
        let old_nodes = self.node_count();
        let old_tick = self.tick;
        let old_rng = self.rng.clone();
        let mut config = self.config;
        config.slices = config.slices.saturating_add(additional_slices);
        let mut expanded = Self::new(config);
        expanded.thermal_state[..old_nodes].copy_from_slice(&self.thermal_state);
        expanded.amplitude[..old_nodes].copy_from_slice(&self.amplitude);
        expanded.phase[..old_nodes].copy_from_slice(&self.phase);
        expanded.temperature[..old_nodes].copy_from_slice(&self.temperature);
        expanded.pilot_force[..old_nodes].copy_from_slice(&self.pilot_force);
        expanded.energy[..old_nodes].copy_from_slice(&self.energy);
        expanded.activation[..old_nodes].copy_from_slice(&self.activation);
        expanded.previous_state[..old_nodes].copy_from_slice(&self.previous_state);
        expanded.previous_phase[..old_nodes].copy_from_slice(&self.previous_phase);
        expanded.tick = old_tick;
        expanded.rng = old_rng;
        let added = expanded.node_count().saturating_sub(old_nodes);
        *self = expanded;
        added
    }

    pub fn replace_edges<I>(&mut self, edges: I)
    where
        I: IntoIterator<Item = (usize, usize, NativeCdtEdgeKind, f32, f32, f32)>,
    {
        self.edge_a.clear();
        self.edge_b.clear();
        self.edge_kind.clear();
        self.edge_weight.clear();
        self.edge_phase.clear();
        self.edge_stability.clear();

        let node_count = self.node_count();
        for (a, b, kind, weight, phase, stability) in edges {
            if a == b || a >= node_count || b >= node_count {
                continue;
            }
            self.edge_a.push(a);
            self.edge_b.push(b);
            self.edge_kind.push(kind);
            self.edge_weight.push(weight.max(0.0));
            self.edge_phase.push(phase);
            self.edge_stability.push(stability.clamp(0.0, 1.0));
        }
        if self.edge_a.is_empty() {
            self.build_foliated_cdt_graph();
        } else {
            self.rebuild_adjacency();
        }
    }

    pub fn compile_sampling_program(&self, config: NativeSamplingConfig) -> NativeSamplingProgram {
        let config = NativeSamplingConfig {
            block_size: config.block_size.max(1),
            schedule_rounds: config.schedule_rounds.max(1),
            max_blocks_per_pulse: config.max_blocks_per_pulse.max(1),
        };
        let mut blocks = Vec::new();
        let mut node_to_block = vec![0; self.node_count()];
        for slice in 0..self.config.slices {
            let slice_start = slice * self.config.nodes_per_slice;
            let slice_end = slice_start + self.config.nodes_per_slice;
            let mut start = slice_start;
            while start < slice_end {
                let end = (start + config.block_size).min(slice_end);
                let id = blocks.len();
                let sampler = match (slice + id) % 3 {
                    0 => NativeSamplerKind::Gaussian,
                    1 => NativeSamplerKind::Gibbs,
                    _ => NativeSamplerKind::Bernoulli,
                };
                let nodes = (start..end).collect::<Vec<_>>();
                for &node in &nodes {
                    node_to_block[node] = id;
                }
                blocks.push(NativeSamplingBlock {
                    id,
                    nodes,
                    sampler,
                    temperature_scale: 1.0 + 0.05 * slice as f32,
                });
                start = end;
            }
        }

        let mut schedule = Vec::with_capacity(blocks.len() * config.schedule_rounds);
        for round in 0..config.schedule_rounds {
            for parity in 0..2 {
                for block in &blocks {
                    if (block.id + round) % 2 == parity {
                        schedule.push(block.id);
                    }
                }
            }
        }

        let mut schedule_rank = vec![usize::MAX; blocks.len()];
        for (rank, &block) in schedule.iter().enumerate() {
            schedule_rank[block] = schedule_rank[block].min(rank);
        }
        NativeSamplingProgram {
            blocks,
            schedule,
            node_to_block,
            schedule_rank,
            config,
        }
    }

    pub fn inject_pilot_pattern(&mut self, nodes: &[usize], amplitude: f32, phase: f32) {
        for &node in nodes {
            if node < self.node_count() {
                self.amplitude[node] = self.amplitude[node].max(amplitude.max(0.0));
                self.phase[node] = phase;
                self.activation[node] = 1.0;
            }
        }
    }

    pub fn local_neighborhood(
        &mut self,
        seeds: &[usize],
        candidates: &[usize],
        max_nodes: usize,
    ) -> Vec<usize> {
        self.neighborhood_generation = self.neighborhood_generation.wrapping_add(1);
        if self.neighborhood_generation == 0 {
            self.neighborhood_marks.fill(0);
            self.neighborhood_generation = 1;
        }
        let generation = self.neighborhood_generation;
        let node_count = self.node_count();
        let limit = max_nodes.min(node_count);
        let mut window = Vec::with_capacity(limit);
        for &node in seeds.iter().chain(candidates.iter()) {
            push_marked_limited(
                &mut window,
                &mut self.neighborhood_marks,
                generation,
                node,
                limit,
            );
            if node >= node_count {
                continue;
            }
            for cursor in self.adjacency_offsets[node]..self.adjacency_offsets[node + 1] {
                let edge = self.adjacency_edges[cursor];
                let other = if self.edge_a[edge] == node {
                    self.edge_b[edge]
                } else {
                    self.edge_a[edge]
                };
                push_marked_limited(
                    &mut window,
                    &mut self.neighborhood_marks,
                    generation,
                    other,
                    limit,
                );
                if window.len() >= limit {
                    return window;
                }
            }
        }
        window
    }

    pub fn pulse_local_pilot(
        &mut self,
        seeds: &[usize],
        candidates: &[usize],
        phase: f32,
        max_nodes: usize,
        microsteps: usize,
    ) -> NativeThermoCdtReport {
        let window = self.local_neighborhood(seeds, candidates, max_nodes);
        for &seed in seeds {
            self.inject_local_node(seed, 2.2, phase, 1.0);
        }
        for &candidate in candidates {
            self.inject_local_node(candidate, 0.65, phase, 0.35);
        }

        let mut report = self.report_local(&window);
        for _ in 0..microsteps {
            report = self.step_local(&window);
        }
        for &node in seeds.iter().chain(candidates.iter()) {
            if node < self.node_count() {
                self.activation[node] = 0.0;
            }
        }
        report
    }

    pub fn pulse_compiled_pilot(
        &mut self,
        program: &NativeSamplingProgram,
        seeds: &[usize],
        candidates: &[usize],
        phase: f32,
        uncertainty: f32,
    ) -> NativeThermoCdtReport {
        let block_ids = program.scheduled_impacted_blocks(seeds, candidates);
        for &seed in seeds {
            self.inject_local_node(seed, 2.2, phase, 1.0);
        }
        for &candidate in candidates {
            self.inject_local_node(candidate, 0.65, phase, 0.35);
        }

        let adaptive_steps = 1
            + (uncertainty.clamp(0.0, 1.0) * program.config.schedule_rounds as f32).ceil() as usize;
        let mut report = NativeThermoCdtReport {
            tick: self.tick,
            nodes: self.node_count(),
            edges: self.edge_count(),
            ..NativeThermoCdtReport::default()
        };

        for _ in 0..adaptive_steps {
            for &block_id in &block_ids {
                let block = &program.blocks[block_id];
                report = self.sample_block(block);
            }
        }

        for &node in seeds.iter().chain(candidates.iter()) {
            if node < self.node_count() {
                self.activation[node] = 0.0;
            }
        }
        report
    }

    pub fn clear_activation(&mut self) {
        self.activation.fill(0.0);
    }

    #[inline(always)]
    pub fn step(&mut self) -> NativeThermoCdtReport {
        self.previous_state.copy_from_slice(&self.thermal_state);
        self.previous_phase.copy_from_slice(&self.phase);

        let dt = self.config.dt.max(0.0);
        let diffusion = self.config.diffusion.max(0.0);
        let confinement = self.config.confinement.max(0.0);
        let pilot_gain = self.config.pilot_gain;
        let phase_coupling = self.config.phase_coupling;
        let amplitude_decay = self.config.amplitude_decay.clamp(0.0, 1.0);
        let state_clamp = self.config.state_clamp;
        let noise_seed = self.rng.gen::<u64>() ^ self.tick.rotate_left(23);
        let previous_state = &self.previous_state;
        let previous_phase = &self.previous_phase;
        let offsets = &self.adjacency_offsets;
        let incident_edges = &self.adjacency_edges;
        let edge_a = &self.edge_a;
        let edge_b = &self.edge_b;
        let edge_weight = &self.edge_weight;
        let edge_phase = &self.edge_phase;

        self.thermal_state
            .par_iter_mut()
            .zip(self.amplitude.par_iter_mut())
            .zip(self.phase.par_iter_mut())
            .zip(self.pilot_force.par_iter_mut())
            .zip(self.energy.par_iter_mut())
            .zip(self.activation.par_iter_mut())
            .zip(self.temperature.par_iter())
            .enumerate()
            .for_each(
                |(
                    node,
                    ((((((state, amplitude), phase), pilot_force), energy), activation), temp),
                )| {
                    let mut laplacian = 0.0;
                    let mut phase_flow = 0.0;
                    for cursor in offsets[node]..offsets[node + 1] {
                        let edge = incident_edges[cursor];
                        let other = if edge_a[edge] == node {
                            edge_b[edge]
                        } else {
                            edge_a[edge]
                        };
                        let weight = edge_weight[edge];
                        laplacian += weight * (previous_state[other] - previous_state[node]);
                        phase_flow += weight
                            * (previous_phase[other] - previous_phase[node] + edge_phase[edge])
                                .sin();
                    }

                    let pilot_potential = *amplitude * phase.sin() + *activation;
                    let force = diffusion * laplacian + pilot_gain * pilot_potential
                        - confinement * previous_state[node];
                    let noise = gaussian_from_counter(noise_seed, node as u64)
                        * (2.0 * temp.max(0.0) * dt).sqrt();
                    let next_state = (previous_state[node] + force * dt + noise)
                        .clamp(-state_clamp, state_clamp);
                    let next_phase = (*phase + phase_coupling * phase_flow * dt + next_state * dt)
                        .rem_euclid(std::f32::consts::TAU);
                    let next_amplitude = (*amplitude * (1.0 - amplitude_decay)
                        + activation.abs() * 0.01)
                        .clamp(0.0, 4.0);

                    *state = next_state;
                    *phase = next_phase;
                    *amplitude = next_amplitude;
                    *pilot_force = force;
                    *energy = effective_energy(next_state, force, confinement, laplacian);
                    *activation *= 0.85;
                },
            );

        self.tick = self.tick.wrapping_add(1);
        self.report()
    }

    pub fn step_local(&mut self, window: &[usize]) -> NativeThermoCdtReport {
        let dt = self.config.dt.max(0.0);
        let diffusion = self.config.diffusion.max(0.0);
        let confinement = self.config.confinement.max(0.0);
        let pilot_gain = self.config.pilot_gain;
        let phase_coupling = self.config.phase_coupling;
        let amplitude_decay = self.config.amplitude_decay.clamp(0.0, 1.0);
        let state_clamp = self.config.state_clamp;
        let noise_seed = self.rng.gen::<u64>() ^ self.tick.rotate_left(23);

        for &node in window {
            if node >= self.node_count() {
                continue;
            }
            let previous_state = self.thermal_state[node];
            let previous_phase = self.phase[node];
            let mut laplacian = 0.0;
            let mut phase_flow = 0.0;
            for cursor in self.adjacency_offsets[node]..self.adjacency_offsets[node + 1] {
                let edge = self.adjacency_edges[cursor];
                let other = if self.edge_a[edge] == node {
                    self.edge_b[edge]
                } else {
                    self.edge_a[edge]
                };
                let weight = self.edge_weight[edge];
                laplacian += weight * (self.thermal_state[other] - previous_state);
                phase_flow +=
                    weight * (self.phase[other] - previous_phase + self.edge_phase[edge]).sin();
            }

            let pilot_potential =
                self.amplitude[node] * self.phase[node].sin() + self.activation[node];
            let force =
                diffusion * laplacian + pilot_gain * pilot_potential - confinement * previous_state;
            let noise = gaussian_from_counter(noise_seed, node as u64)
                * (2.0 * self.temperature[node].max(0.0) * dt).sqrt();
            let next_state = (previous_state + force * dt + noise).clamp(-state_clamp, state_clamp);

            self.thermal_state[node] = next_state;
            self.phase[node] =
                (self.phase[node] + phase_coupling * phase_flow * dt + next_state * dt)
                    .rem_euclid(std::f32::consts::TAU);
            self.amplitude[node] = (self.amplitude[node] * (1.0 - amplitude_decay)
                + self.activation[node].abs() * 0.01)
                .clamp(0.0, 4.0);
            self.pilot_force[node] = force;
            self.energy[node] = effective_energy(next_state, force, confinement, laplacian);
            self.activation[node] *= 0.35;
        }

        self.tick = self.tick.wrapping_add(1);
        self.report_local(window)
    }

    pub fn sample_block(&mut self, block: &NativeSamplingBlock) -> NativeThermoCdtReport {
        let noise_seed = self.rng.gen::<u64>() ^ self.tick.rotate_left(29) ^ block.id as u64;
        for (offset, &node) in block.nodes.iter().enumerate() {
            if node >= self.node_count() {
                continue;
            }
            match block.sampler {
                NativeSamplerKind::Gaussian => {
                    self.sample_gaussian_node(
                        node,
                        noise_seed,
                        offset as u64,
                        block.temperature_scale,
                    );
                }
                NativeSamplerKind::Gibbs => {
                    self.sample_gibbs_node(
                        node,
                        noise_seed,
                        offset as u64,
                        block.temperature_scale,
                    );
                }
                NativeSamplerKind::Bernoulli => {
                    self.sample_bernoulli_node(
                        node,
                        noise_seed,
                        offset as u64,
                        block.temperature_scale,
                    );
                }
            }
        }
        self.tick = self.tick.wrapping_add(1);
        self.block_observables(block)
            .into_report(self.tick, self.node_count(), self.edge_count())
    }

    pub fn run_until_stable(
        &mut self,
        max_steps: usize,
        energy_tolerance: f32,
        variance_tolerance: f32,
    ) -> NativeThermoCdtReport {
        let mut previous = self.report();
        let mut current = previous;
        for _ in 0..max_steps {
            current = self.step();
            if (current.mean_energy - previous.mean_energy).abs() <= energy_tolerance
                && (current.state_variance - previous.state_variance).abs() <= variance_tolerance
            {
                break;
            }
            previous = current;
        }
        current
    }

    pub fn report(&self) -> NativeThermoCdtReport {
        const PARALLEL_REPORT_THRESHOLD: usize = 8_192;
        let stats = if self.node_count() < PARALLEL_REPORT_THRESHOLD {
            (0..self.node_count()).fold(ThermalStats::default(), |stats, node| {
                stats.merge(self.node_stats(node))
            })
        } else {
            (0..self.node_count())
                .into_par_iter()
                .map(|node| self.node_stats(node))
                .reduce(ThermalStats::default, ThermalStats::merge)
        };
        let n = stats.count.max(1) as f64;
        let mean_state = stats.state_sum / n;
        let state_variance = (stats.state_sq_sum / n - mean_state * mean_state).max(0.0);
        let partition = stats.partition_sum.max(EPSILON as f64);

        NativeThermoCdtReport {
            tick: self.tick,
            nodes: self.node_count(),
            edges: self.edge_count(),
            mean_state: mean_state as f32,
            state_variance: state_variance as f32,
            mean_amplitude: (stats.amplitude_sum / n) as f32,
            mean_energy: (stats.energy_sum / n) as f32,
            free_energy_proxy: -self.config.temperature.max(EPSILON) * partition.ln() as f32,
            active_nodes: stats.active_nodes,
            max_abs_state: stats.max_abs_state,
        }
    }

    pub fn report_local(&self, window: &[usize]) -> NativeThermoCdtReport {
        let mut stats = ThermalStats::default();
        for &node in window.iter().filter(|&&node| node < self.node_count()) {
            stats = stats.merge(self.node_stats(node));
        }
        let n = stats.count.max(1) as f64;
        let mean_state = stats.state_sum / n;
        let state_variance = (stats.state_sq_sum / n - mean_state * mean_state).max(0.0);

        NativeThermoCdtReport {
            tick: self.tick,
            nodes: self.node_count(),
            edges: self.edge_count(),
            mean_state: mean_state as f32,
            state_variance: state_variance as f32,
            mean_amplitude: (stats.amplitude_sum / n) as f32,
            mean_energy: (stats.energy_sum / n) as f32,
            free_energy_proxy: -self.config.temperature.max(EPSILON)
                * stats.partition_sum.max(EPSILON as f64).ln() as f32,
            active_nodes: stats.active_nodes,
            max_abs_state: stats.max_abs_state,
        }
    }

    #[inline(always)]
    fn node_stats(&self, node: usize) -> ThermalStats {
        let state = self.thermal_state[node];
        let energy = self.energy[node];
        ThermalStats {
            count: 1,
            state_sum: state as f64,
            state_sq_sum: (state as f64) * (state as f64),
            amplitude_sum: self.amplitude[node] as f64,
            energy_sum: energy as f64,
            partition_sum: (-energy / self.temperature[node].max(EPSILON)).exp() as f64,
            active_nodes: usize::from(self.activation[node] > 0.05),
            max_abs_state: state.abs(),
        }
    }

    /// Inyección local (pilot). Pisa fase y toma max de amplitud/activación.
    pub fn inject_local_node(&mut self, node: usize, amplitude: f32, phase: f32, activation: f32) {
        if node < self.node_count() {
            self.amplitude[node] = self.amplitude[node].max(amplitude.max(0.0));
            self.phase[node] = phase;
            self.activation[node] = self.activation[node].max(activation.max(0.0));
        }
    }

    /// Inyección débil tipo energía de vacío: mezcla fase, no fuerza colapso.
    pub fn inject_vacuum_node(&mut self, node: usize, amplitude: f32, phase: f32, activation: f32) {
        if node < self.node_count() {
            let activation = activation.max(0.0);
            let amount = activation.clamp(0.0, 1.0);
            self.amplitude[node] = self.amplitude[node].max(amplitude.max(0.0));
            let cur = self.phase[node];
            let sin = (1.0 - amount) * cur.sin() + amount * phase.sin();
            let cos = (1.0 - amount) * cur.cos() + amount * phase.cos();
            self.phase[node] = sin.atan2(cos).rem_euclid(std::f32::consts::TAU);
            self.activation[node] = self.activation[node].max(activation);
        }
    }

    fn sample_gaussian_node(
        &mut self,
        node: usize,
        noise_seed: u64,
        offset: u64,
        temperature_scale: f32,
    ) {
        let (force, laplacian) = self.local_force(node);
        let dt = self.config.dt.max(0.0);
        let noise = gaussian_from_counter(noise_seed, offset)
            * (2.0 * self.temperature[node].max(0.0) * temperature_scale * dt).sqrt();
        self.commit_node(node, force, laplacian, force * dt + noise);
    }

    fn sample_gibbs_node(
        &mut self,
        node: usize,
        noise_seed: u64,
        offset: u64,
        temperature_scale: f32,
    ) {
        let (force, laplacian) = self.local_force(node);
        let temp = (self.temperature[node] * temperature_scale).max(EPSILON);
        let proposal = (force / temp).tanh();
        let jitter = 0.05 * gaussian_from_counter(noise_seed, offset);
        self.commit_node(
            node,
            force,
            laplacian,
            proposal - self.thermal_state[node] + jitter,
        );
    }

    fn sample_bernoulli_node(
        &mut self,
        node: usize,
        noise_seed: u64,
        offset: u64,
        temperature_scale: f32,
    ) {
        let (force, laplacian) = self.local_force(node);
        let temp = (self.temperature[node] * temperature_scale).max(EPSILON);
        let probability = sigmoid(force / temp);
        let draw = unit_from_u64(splitmix64(
            noise_seed ^ offset.wrapping_mul(0xA24B_AED4_963E_E407),
        ));
        let target = if draw < probability { 1.0 } else { -1.0 };
        self.commit_node(
            node,
            force,
            laplacian,
            0.25 * (target - self.thermal_state[node]),
        );
    }

    fn local_force(&self, node: usize) -> (f32, f32) {
        let mut laplacian = 0.0;
        let mut phase_flow = 0.0;
        for cursor in self.adjacency_offsets[node]..self.adjacency_offsets[node + 1] {
            let edge = self.adjacency_edges[cursor];
            let other = if self.edge_a[edge] == node {
                self.edge_b[edge]
            } else {
                self.edge_a[edge]
            };
            let weight = self.edge_weight[edge];
            laplacian += weight * (self.thermal_state[other] - self.thermal_state[node]);
            phase_flow +=
                weight * (self.phase[other] - self.phase[node] + self.edge_phase[edge]).sin();
        }
        let pilot_potential = self.amplitude[node] * self.phase[node].sin() + self.activation[node];
        let force = self.config.diffusion.max(0.0) * laplacian
            + self.config.pilot_gain * pilot_potential
            + self.config.phase_coupling * phase_flow
            - self.config.confinement.max(0.0) * self.thermal_state[node];
        (force, laplacian)
    }

    fn commit_node(&mut self, node: usize, force: f32, laplacian: f32, delta: f32) {
        let next_state = (self.thermal_state[node] + delta)
            .clamp(-self.config.state_clamp, self.config.state_clamp);
        self.thermal_state[node] = next_state;
        self.phase[node] =
            (self.phase[node] + next_state * self.config.dt).rem_euclid(std::f32::consts::TAU);
        self.amplitude[node] = (self.amplitude[node]
            * (1.0 - self.config.amplitude_decay.clamp(0.0, 1.0))
            + self.activation[node].abs() * 0.01)
            .clamp(0.0, 4.0);
        self.pilot_force[node] = force;
        self.energy[node] = effective_energy(next_state, force, self.config.confinement, laplacian);
        self.activation[node] *= 0.35;
    }

    pub fn block_observables(&self, block: &NativeSamplingBlock) -> NativeBlockObservables {
        let mut obs = NativeBlockObservables {
            block_id: block.id,
            nodes: block.nodes.len(),
            ..NativeBlockObservables::default()
        };
        if block.nodes.is_empty() {
            return obs;
        }
        for &node in &block.nodes {
            obs.mean_state += self.thermal_state[node];
            obs.mean_energy += self.energy[node];
            obs.mean_activation += self.activation[node];
        }
        let n = block.nodes.len() as f32;
        obs.mean_state /= n;
        obs.mean_energy /= n;
        obs.mean_activation /= n;
        for &node in &block.nodes {
            let centered = self.thermal_state[node] - obs.mean_state;
            obs.state_variance += centered * centered;
        }
        obs.state_variance /= n;
        obs.uncertainty = (obs.state_variance.sqrt() + obs.mean_activation.abs()).clamp(0.0, 1.0);
        obs
    }

    fn build_foliated_cdt_graph(&mut self) {
        for slice in 0..self.config.slices {
            for offset in 0..self.config.nodes_per_slice {
                let node = self.node_id(slice, offset);
                for hop in 1..=self.config.spatial_degree {
                    let other = self.node_id(slice, (offset + hop) % self.config.nodes_per_slice);
                    self.add_edge(node, other, NativeCdtEdgeKind::Spatial, 1.0 / hop as f32);
                }
                if slice + 1 < self.config.slices {
                    for hop in 0..self.config.temporal_degree {
                        let shifted = (offset + hop) % self.config.nodes_per_slice;
                        let other = self.node_id(slice + 1, shifted);
                        self.add_edge(node, other, NativeCdtEdgeKind::Temporal, 0.85);
                    }
                }
            }
        }
        self.rebuild_adjacency();
    }

    fn node_id(&self, slice: usize, offset: usize) -> usize {
        slice * self.config.nodes_per_slice + offset
    }

    fn add_edge(&mut self, a: usize, b: usize, kind: NativeCdtEdgeKind, weight: f32) {
        if a == b {
            return;
        }
        let edge = self.edge_a.len();
        self.edge_a.push(a);
        self.edge_b.push(b);
        self.edge_kind.push(kind);
        self.edge_weight.push(weight);
        self.edge_phase.push(match kind {
            NativeCdtEdgeKind::Spatial => 0.0,
            NativeCdtEdgeKind::Temporal => 0.25,
        });
        self.edge_stability.push(1.0 - 0.001 * edge as f32);
    }

    fn rebuild_adjacency(&mut self) {
        let nodes = self.node_count();
        let mut degree = vec![0_usize; nodes];
        for (&a, &b) in self.edge_a.iter().zip(&self.edge_b) {
            degree[a] += 1;
            degree[b] += 1;
        }

        self.adjacency_offsets = vec![0; nodes + 1];
        for node in 0..nodes {
            self.adjacency_offsets[node + 1] = self.adjacency_offsets[node] + degree[node];
        }

        self.adjacency_edges = vec![0; self.adjacency_offsets[nodes]];
        let mut cursor = self.adjacency_offsets[..nodes].to_vec();
        for edge in 0..self.edge_a.len() {
            let a = self.edge_a[edge];
            let b = self.edge_b[edge];
            self.adjacency_edges[cursor[a]] = edge;
            cursor[a] += 1;
            self.adjacency_edges[cursor[b]] = edge;
            cursor[b] += 1;
        }
    }
}

impl NativeSamplingProgram {
    pub fn impacted_blocks(&self, seeds: &[usize], candidates: &[usize]) -> Vec<usize> {
        let mut blocks = Vec::with_capacity(self.config.max_blocks_per_pulse);
        // Las semillas tienen prioridad absoluta y conservan su orden.
        for &node in seeds {
            if node >= self.node_to_block.len() {
                continue;
            }
            let block = self.node_to_block[node];
            push_unique_limited(
                &mut blocks,
                block,
                self.config.max_blocks_per_pulse,
                self.blocks.len(),
            );
            if blocks.len() >= self.config.max_blocks_per_pulse {
                return blocks;
            }
        }
        // Antes se ordenaban todos los candidate ids para escoger sus primeros
        // bloques. Mantener un top-K por id mínimo produce el mismo conjunto sin
        // ordenar el vector completo.
        let remaining = self.config.max_blocks_per_pulse - blocks.len();
        let mut candidate_blocks = Vec::<(usize, usize)>::with_capacity(remaining);
        for &node in candidates {
            if node >= self.node_to_block.len() {
                continue;
            }
            let block = self.node_to_block[node];
            if blocks.contains(&block) {
                continue;
            }
            if let Some((_, min_node)) = candidate_blocks
                .iter_mut()
                .find(|(existing, _)| *existing == block)
            {
                *min_node = (*min_node).min(node);
                continue;
            }
            if candidate_blocks.len() < remaining {
                candidate_blocks.push((block, node));
                continue;
            }
            if let Some((worst_index, _)) = candidate_blocks
                .iter()
                .enumerate()
                .max_by_key(|(_, (_, min_node))| *min_node)
            {
                if node < candidate_blocks[worst_index].1 {
                    candidate_blocks[worst_index] = (block, node);
                }
            }
        }
        candidate_blocks.sort_unstable_by_key(|(_, min_node)| *min_node);
        blocks.extend(candidate_blocks.into_iter().map(|(block, _)| block));
        blocks
    }

    pub fn scheduled_impacted_blocks(&self, seeds: &[usize], candidates: &[usize]) -> Vec<usize> {
        let mut impacted = self.impacted_blocks(seeds, candidates);
        impacted.sort_by_key(|&block| self.schedule_rank[block]);
        impacted
    }
}

impl NativeBlockObservables {
    fn into_report(self, tick: u64, nodes: usize, edges: usize) -> NativeThermoCdtReport {
        NativeThermoCdtReport {
            tick,
            nodes,
            edges,
            mean_state: self.mean_state,
            state_variance: self.state_variance,
            mean_amplitude: 0.0,
            mean_energy: self.mean_energy,
            free_energy_proxy: -self.mean_energy,
            active_nodes: usize::from(self.mean_activation > 0.05) * self.nodes,
            max_abs_state: self.mean_state.abs() + self.state_variance.sqrt(),
        }
    }
}

#[inline(always)]
fn effective_energy(state: f32, force: f32, confinement: f32, laplacian: f32) -> f32 {
    0.5 * confinement * state * state - force * state + 0.5 * laplacian * laplacian
}

#[inline(always)]
fn gaussian_from_counter(seed: u64, counter: u64) -> f32 {
    let base = seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let sum = unit_from_u64(splitmix64(base))
        + unit_from_u64(splitmix64(base ^ 0xA24B_AED4_963E_E407))
        + unit_from_u64(splitmix64(base ^ 0x9FB2_1C65_1E98_DF25))
        + unit_from_u64(splitmix64(base ^ 0xC13F_A9A9_02A6_328F))
        + unit_from_u64(splitmix64(base ^ 0x91E1_0DA5_C79E_7B1D))
        + unit_from_u64(splitmix64(base ^ 0xD1B5_4A32_D192_ED03));
    sum - 3.0
}

#[inline(always)]
fn unit_from_u64(value: u64) -> f32 {
    ((value >> 40) as f32) * (1.0 / (1_u32 << 24) as f32)
}

#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn push_unique_limited(out: &mut Vec<usize>, node: usize, max_nodes: usize, node_count: usize) {
    if out.len() >= max_nodes || node >= node_count || out.contains(&node) {
        return;
    }
    out.push(node);
}

#[inline(always)]
fn push_marked_limited(
    out: &mut Vec<usize>,
    marks: &mut [u32],
    generation: u32,
    node: usize,
    max_nodes: usize,
) {
    if out.len() >= max_nodes || node >= marks.len() || marks[node] == generation {
        return;
    }
    marks[node] = generation;
    out.push(node);
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_cdt_builds_flat_graph() {
        let substrate = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 3,
            nodes_per_slice: 16,
            spatial_degree: 2,
            temporal_degree: 1,
            ..NativeThermoCdtConfig::default()
        });

        assert_eq!(substrate.node_count(), 48);
        assert!(substrate.edge_count() > substrate.node_count());
        assert_eq!(
            substrate.adjacency_offsets.len(),
            substrate.node_count() + 1
        );
    }

    #[test]
    fn native_cdt_grows_without_moving_existing_state() {
        let mut substrate = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice: 8,
            ..NativeThermoCdtConfig::default()
        });
        substrate.thermal_state[3] = 1.25;
        substrate.amplitude[3] = 0.75;
        assert_eq!(substrate.grow_slices(2), 16);
        assert_eq!(substrate.node_count(), 32);
        assert_eq!(substrate.config.slices, 4);
        assert_eq!(substrate.thermal_state[3], 1.25);
        assert_eq!(substrate.amplitude[3], 0.75);
        assert_eq!(substrate.adjacency_offsets.len(), 33);
    }

    #[test]
    fn native_cdt_evolves_without_diverging() {
        let mut substrate = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice: 32,
            ..NativeThermoCdtConfig::default()
        });

        substrate.inject_pilot_pattern(&[0, 1, 2, 33], 2.0, 0.0);
        let report = substrate.run_until_stable(8, 1.0e-4, 1.0e-4);

        assert_eq!(report.nodes, 64);
        assert_eq!(report.tick, substrate.tick());
        assert!(report.max_abs_state <= substrate.config.state_clamp);
        assert!(report.mean_energy.is_finite());
    }

    #[test]
    fn compiled_sampling_program_updates_impacted_blocks() {
        let mut substrate = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice: 24,
            ..NativeThermoCdtConfig::default()
        });
        let program = substrate.compile_sampling_program(NativeSamplingConfig {
            block_size: 8,
            schedule_rounds: 1,
            max_blocks_per_pulse: 3,
        });

        let report = substrate.pulse_compiled_pilot(&program, &[0, 1], &[25, 26], 0.0, 0.75);

        assert!(!program.blocks.is_empty());
        assert!(!program.schedule.is_empty());
        assert!(report.mean_energy.is_finite());
        assert!(report.max_abs_state <= substrate.config.state_clamp);
    }
}
