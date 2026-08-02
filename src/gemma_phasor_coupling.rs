//! Acople concurrente entre el decode Gemma 2 y la inferencia fasorial.
//!
//! Candle permanece en el hilo de generación. Un worker Rust independiente
//! recibe tokens por canal y evoluciona su propio motor fasorial, de modo que
//! la termodinámica no bloquea la latencia por token del LLM.

use crate::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorReport, NativePhasorThermodynamicEngine,
};
use crate::native_rng::splitmix64;
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Clone, Copy, Debug)]
pub struct GemmaPhasorCouplingConfig {
    pub nodes: usize,
    pub spatial_degree: usize,
    pub active_nodes_per_token: usize,
    pub steps_per_token: usize,
    /// Agrupa estímulos antes de evolucionar el campo para no competir con
    /// cada matmul del decode en CPU.
    pub step_every_tokens: usize,
    pub stimulus_amplitude: f32,
    pub seed: u64,
}

impl Default for GemmaPhasorCouplingConfig {
    fn default() -> Self {
        Self {
            nodes: 256,
            spatial_degree: 4,
            active_nodes_per_token: 8,
            steps_per_token: 1,
            step_every_tokens: 16,
            stimulus_amplitude: 0.18,
            seed: 0x4745_4D4D_4150_4841,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GemmaPhasorStreamReport {
    pub observed_tokens: u64,
    pub phasor_steps: u64,
    pub state: NativePhasorReport,
}

enum Command {
    Token { token: u32, position: usize },
    Snapshot(Sender<GemmaPhasorStreamReport>),
    Shutdown,
}

pub struct GemmaPhasorWorker {
    sender: Sender<Command>,
    handle: Option<JoinHandle<()>>,
}

impl GemmaPhasorWorker {
    pub fn start(config: GemmaPhasorCouplingConfig) -> Result<Self, String> {
        let engine = NativePhasorThermodynamicEngine::from_cdt_config(
            NativeThermoCdtConfig {
                slices: 1,
                nodes_per_slice: config.nodes.max(1),
                spatial_degree: config.spatial_degree,
                temporal_degree: 1,
                temperature: 0.0,
                seed: config.seed,
                ..NativeThermoCdtConfig::default()
            },
            NativePhasorConfig {
                temperature_scale: 0.0,
                noise_scale: 0.0,
                stimulus_decay: 0.92,
                seed: config.seed ^ 0x5048_4153_4F52,
                ..NativePhasorConfig::default()
            },
        )
        .map_err(|error| error.to_string())?;
        let (sender, receiver) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("gemma-phasor-stream".to_string())
            .spawn(move || worker_loop(engine, config, receiver))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            sender,
            handle: Some(handle),
        })
    }

    /// Encola un token sin mover tensores Candle entre hilos.
    pub fn observe_token(&self, token: u32, position: usize) {
        let _ = self.sender.send(Command::Token { token, position });
    }

    pub fn snapshot(&self, timeout: Duration) -> Option<GemmaPhasorStreamReport> {
        let (sender, receiver) = mpsc::channel();
        self.sender.send(Command::Snapshot(sender)).ok()?;
        receiver.recv_timeout(timeout).ok()
    }
}

impl Drop for GemmaPhasorWorker {
    fn drop(&mut self) {
        let _ = self.sender.send(Command::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(
    mut engine: NativePhasorThermodynamicEngine,
    config: GemmaPhasorCouplingConfig,
    receiver: Receiver<Command>,
) {
    let mut observed_tokens = 0u64;
    let mut phasor_steps = 0u64;
    while let Ok(command) = receiver.recv() {
        match command {
            Command::Token { token, position } => {
                let nodes = token_nodes(
                    token,
                    position,
                    engine.node_count(),
                    config.active_nodes_per_token,
                    config.seed,
                );
                let phase = (token as f32 * 0.618_034).fract() * std::f32::consts::TAU;
                engine.inject_pattern(&nodes, config.stimulus_amplitude, phase);
                observed_tokens = observed_tokens.saturating_add(1);
                if observed_tokens.is_multiple_of(config.step_every_tokens.max(1) as u64) {
                    for _ in 0..config.steps_per_token {
                        engine.step();
                        phasor_steps = phasor_steps.saturating_add(1);
                    }
                }
            }
            Command::Snapshot(reply) => {
                let _ = reply.send(GemmaPhasorStreamReport {
                    observed_tokens,
                    phasor_steps,
                    state: engine.report(),
                });
            }
            Command::Shutdown => break,
        }
    }
}

fn token_nodes(
    token: u32,
    position: usize,
    node_count: usize,
    limit: usize,
    seed: u64,
) -> Vec<usize> {
    let mut state = seed ^ (token as u64) ^ (position as u64).rotate_left(23);
    let mut nodes = Vec::with_capacity(limit);
    for _ in 0..limit.min(node_count) {
        state = splitmix64(state);
        let mut node = state as usize % node_count.max(1);
        while nodes.contains(&node) {
            node = (node + 1) % node_count.max(1);
        }
        nodes.push(node);
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_worker_processes_token_stream() {
        let worker = GemmaPhasorWorker::start(GemmaPhasorCouplingConfig {
            nodes: 64,
            spatial_degree: 2,
            active_nodes_per_token: 4,
            step_every_tokens: 1,
            ..GemmaPhasorCouplingConfig::default()
        })
        .unwrap();
        for (position, token) in [7, 11, 13, 17].into_iter().enumerate() {
            worker.observe_token(token, position);
        }
        let report = worker.snapshot(Duration::from_secs(2)).unwrap();
        assert_eq!(report.observed_tokens, 4);
        assert_eq!(report.phasor_steps, 4);
        assert_eq!(report.state.nodes, 64);
        assert!(report.state.free_energy.is_finite());
    }

    #[test]
    fn token_projection_is_deterministic_and_unique() {
        let first = token_nodes(42, 3, 128, 8, 9);
        let second = token_nodes(42, 3, 128, 8, 9);
        assert_eq!(first, second);
        let mut unique = first.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), first.len());
    }
}
