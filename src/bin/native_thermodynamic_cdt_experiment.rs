use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use std::mem;
use std::time::Instant;

const STEPS: usize = 512;
const PATTERN_PERIOD: usize = 16;

fn main() {
    let mut substrate = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: 6,
        nodes_per_slice: 256,
        spatial_degree: 4,
        temporal_degree: 2,
        temperature: 0.28,
        dt: 0.01,
        diffusion: 0.22,
        confinement: 0.05,
        pilot_gain: 0.48,
        phase_coupling: 0.18,
        amplitude_decay: 0.003,
        state_clamp: 4.0,
        seed: 0xCD7A_71C0,
    });

    let initial = substrate.report();
    let start = Instant::now();
    let mut report = initial;
    for step in 0..STEPS {
        if step % PATTERN_PERIOD == 0 {
            let pattern = pilot_pattern(step / PATTERN_PERIOD, substrate.node_count());
            substrate.inject_pilot_pattern(&pattern, 2.4, step as f32 * 0.05);
        }
        report = substrate.step();
    }
    let elapsed = start.elapsed();
    let memory_kib = estimated_memory_kib(&substrate);
    let node_steps = substrate.node_count() as f64 * STEPS as f64;
    let mega_node_steps_per_sec = node_steps / elapsed.as_secs_f64() / 1_000_000.0;

    println!("Native thermodynamic CDT substrate experiment");
    println!(
        "config: nodes={} edges={} steps={} slices={} nodes_per_slice={}",
        substrate.node_count(),
        substrate.edge_count(),
        STEPS,
        substrate.config.slices,
        substrate.config.nodes_per_slice
    );
    println!(
        "initial: mean_state={:.5} variance={:.5} mean_amp={:.5} mean_energy={:.5} free_energy={:.5}",
        initial.mean_state,
        initial.state_variance,
        initial.mean_amplitude,
        initial.mean_energy,
        initial.free_energy_proxy
    );
    println!(
        "final: tick={} mean_state={:.5} variance={:.5} mean_amp={:.5} mean_energy={:.5} free_energy={:.5} active_nodes={} max_abs_state={:.5}",
        report.tick,
        report.mean_state,
        report.state_variance,
        report.mean_amplitude,
        report.mean_energy,
        report.free_energy_proxy,
        report.active_nodes,
        report.max_abs_state
    );
    println!(
        "performance: elapsed_ms={:.3} us_per_step={:.3} mega_node_steps_per_sec={:.3} estimated_memory_kib={:.1}",
        elapsed.as_secs_f64() * 1_000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / STEPS as f64,
        mega_node_steps_per_sec,
        memory_kib
    );
}

fn pilot_pattern(seed: usize, nodes: usize) -> Vec<usize> {
    let stride = 37 + seed % 11;
    (0..12)
        .map(|idx| (seed * 97 + idx * stride) % nodes.max(1))
        .collect()
}

fn estimated_memory_kib(substrate: &NativeThermoCdtSubstrate) -> f32 {
    let node_count = substrate.node_count();
    let edge_count = substrate.edge_count();
    let node_arrays = 7 * node_count * mem::size_of::<f32>();
    let edge_arrays = edge_count
        * (2 * mem::size_of::<usize>()
            + mem::size_of::<cdt_rqm_epr::native_thermodynamic_cdt::NativeCdtEdgeKind>()
            + 3 * mem::size_of::<f32>());
    let scratch = 2 * node_count * mem::size_of::<f32>();
    (node_arrays + edge_arrays + scratch) as f32 / 1024.0
}
