//! A/B nativo y pareado: Armijo frente a Handshake sobre la misma F.

use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorInferencePolicy, NativePhasorMinimizationReport,
    NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
};
use cdt_rqm_epr::native_rng::{splitmix64, unit_from_u64};
use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use num_complex::Complex32;
use std::time::{Duration, Instant};

const TRIALS: usize = 12;
const ITERATIONS: usize = 32;

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    elapsed: Duration,
    free_energy_per_node: f64,
    residual: f64,
    coherence: f64,
    target_alignment: f64,
    evaluations: usize,
    accepted: usize,
    handshake_applications: usize,
    integrated_information: f64,
    attention_ignitions: usize,
    handshake_iterations: usize,
    attention_probes: usize,
    release_iteration: usize,
    energy_wins: usize,
    residual_wins: usize,
    alignment_wins: usize,
    cases: usize,
}

impl Metrics {
    fn record(
        &mut self,
        engine: &NativePhasorThermodynamicEngine,
        result: NativePhasorMinimizationReport,
        target: &[Complex32],
        elapsed: Duration,
    ) {
        let report = result.final_report;
        self.elapsed += elapsed;
        self.free_energy_per_node += f64::from(report.free_energy) / report.nodes.max(1) as f64;
        self.residual += f64::from(report.gradient_residual);
        self.coherence += f64::from(report.phase_coherence);
        self.target_alignment += f64::from(target_alignment(&engine.phasors, target));
        self.evaluations += result.energy_evaluations;
        self.accepted += result.accepted_steps;
        self.handshake_applications += result.handshake_operator_applications;
        self.integrated_information += f64::from(result.mean_integrated_information);
        self.attention_ignitions += result.attention_ignitions;
        self.handshake_iterations += result.handshake_iterations;
        self.attention_probes += result.attention_probes;
        self.release_iteration += result.modifier_release_iteration;
        self.cases += 1;
    }

    fn accumulate(&mut self, other: &Self) {
        self.elapsed += other.elapsed;
        self.energy_wins += other.energy_wins;
        self.residual_wins += other.residual_wins;
        self.alignment_wins += other.alignment_wins;
        self.cases += other.cases;
    }

    fn print(self, nodes: usize, method: &str) {
        let cases = self.cases.max(1) as f64;
        println!(
            "{nodes},{method},{:.3},{:.7},{:.3e},{:.6},{:.6},{:.2},{:.2},{:.2},{:.6},{:.2},\
             {:.2},{:.2},{:.2},{},{},{}",
            self.elapsed.as_secs_f64() * 1_000.0 / cases,
            self.free_energy_per_node / cases,
            self.residual / cases,
            self.coherence / cases,
            self.target_alignment / cases,
            self.evaluations as f64 / cases,
            self.accepted as f64 / cases,
            self.handshake_applications as f64 / cases,
            self.integrated_information / cases,
            self.attention_ignitions as f64 / cases,
            self.handshake_iterations as f64 / cases,
            self.attention_probes as f64 / cases,
            self.release_iteration as f64 / cases,
            self.energy_wins,
            self.residual_wins,
            self.alignment_wins,
        );
    }
}

/// Cada variante mueve sólo la programación de los moduladores. F, topología,
/// estado inicial, estímulo, semilla y presupuesto de iteraciones son idénticos.
const VARIANTS: [(&str, f32, f32, NativePhasorInferencePolicy); 4] = [
    ("armijo", 0.0, 0.0, NativePhasorInferencePolicy::Fixed),
    (
        "handshake_armijo",
        0.65,
        0.0,
        NativePhasorInferencePolicy::Fixed,
    ),
    (
        "handshake_atencion_fijo",
        0.65,
        0.55,
        NativePhasorInferencePolicy::Fixed,
    ),
    (
        "hibrido_adaptativo",
        0.65,
        0.55,
        NativePhasorInferencePolicy::Adaptive,
    ),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("benchmark=armijo_vs_handshake_motor_fasorial_nativo");
    println!("control=misma_topologia_estado_estimulo_F_semilla_iteraciones");
    println!("handshake=precondicionador_adjunto no_retrocausalidad");
    println!("hibrido=handshake_hasta_saturar+atencion_por_ignicion+cola_armijo_pura");
    println!("trials={TRIALS} iterations={ITERATIONS}");
    println!(
        "nodes,method,mean_ms,F_per_node,residual,coherence,target_alignment,\
         energy_evaluations,accepted,backward_operator_applications,phi,attention_ignitions,\
         handshake_iterations,attention_probes,release_iteration,\
         energy_wins,residual_wins,alignment_wins"
    );

    let mut totals = [Metrics::default(); VARIANTS.len()];
    let mut comparisons = 0usize;

    for nodes in [128usize, 512, 2_048] {
        let template = fixture(nodes)?;
        let mut metrics = [Metrics::default(); VARIANTS.len()];

        for trial in 0..TRIALS {
            let seed = 0x4841_4E44_2026_u64 ^ nodes as u64 ^ (trial as u64).rotate_left(29);
            let mut initial = template.clone();
            let target = initialize_paired_problem(&mut initial, seed);

            let mut baseline_energy = 0.0f32;
            let mut baseline_residual = 0.0f32;
            let mut baseline_alignment = 0.0f32;

            for (index, (name, handshake, attention, policy)) in VARIANTS.iter().enumerate() {
                let mut engine = initial.clone();
                let started = Instant::now();
                let result = engine.minimize_free_energy(config(*handshake, *attention, *policy));
                let elapsed = started.elapsed();

                if result.final_report.free_energy > result.initial.free_energy + 1.0e-5 {
                    return Err(format!("{name} violó la aceptación monótona de Armijo").into());
                }
                let alignment = target_alignment(&engine.phasors, &target);
                if index == 0 {
                    baseline_energy = result.final_report.free_energy;
                    baseline_residual = result.final_report.gradient_residual;
                    baseline_alignment = alignment;
                } else {
                    metrics[index].energy_wins +=
                        usize::from(result.final_report.free_energy < baseline_energy);
                    metrics[index].residual_wins +=
                        usize::from(result.final_report.gradient_residual < baseline_residual);
                    metrics[index].alignment_wins += usize::from(alignment > baseline_alignment);
                }
                metrics[index].record(&engine, result, &target, elapsed);
            }
            comparisons += 1;
        }

        for (index, (name, ..)) in VARIANTS.iter().enumerate() {
            totals[index].accumulate(&metrics[index]);
            metrics[index].print(nodes, name);
        }
    }

    println!("comparaciones_pareadas={comparisons}");
    for (index, (name, ..)) in VARIANTS.iter().enumerate().skip(1) {
        let speed = totals[index].elapsed.as_secs_f64() / totals[0].elapsed.as_secs_f64().max(1e-9);
        println!(
            "resumen,{name},energy_wins={},residual_wins={},alignment_wins={},costo_relativo={speed:.2}x",
            totals[index].energy_wins,
            totals[index].residual_wins,
            totals[index].alignment_wins,
        );
    }
    Ok(())
}

fn config(
    handshake_strength: f32,
    attention_strength: f32,
    inference_policy: NativePhasorInferencePolicy,
) -> NativePhasorMinimizerConfig {
    NativePhasorMinimizerConfig {
        max_iterations: ITERATIONS,
        energy_tolerance: 0.0,
        residual_tolerance: 0.0,
        topological_warm_start: false,
        attention_strength,
        attention_temperature: 0.75,
        attention_max_gain: 3.0,
        attention_ignition_threshold: 0.001,
        handshake_strength,
        handshake_rounds: 4,
        handshake_damping: 0.25,
        handshake_max_gain: 3.0,
        inference_policy,
        ..NativePhasorMinimizerConfig::default()
    }
}

fn fixture(nodes: usize) -> Result<NativePhasorThermodynamicEngine, Box<dyn std::error::Error>> {
    let mut core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: 1,
        nodes_per_slice: nodes,
        spatial_degree: 4,
        temporal_degree: 1,
        temperature: 0.0,
        seed: 0xCD70_0000 ^ nodes as u64,
        ..NativeThermoCdtConfig::default()
    });
    // Atractor conocido y compatible con el grafo: transporte de fase nulo.
    // Así la frontera parcial puede propagarse sin entregar la solución.
    core.edge_phase.fill(0.0);
    Ok(NativePhasorThermodynamicEngine::from_core(
        &core,
        NativePhasorConfig {
            coupling_strength: 1.0,
            radial_strength: 3.0,
            target_amplitude: 1.0,
            confinement: 0.02,
            stimulus_gain: 0.9,
            stimulus_decay: 1.0,
            entropy_weight: 0.0,
            temperature_scale: 0.0,
            noise_scale: 0.0,
            max_amplitude: 2.0,
            ..NativePhasorConfig::default()
        },
    )?)
}

fn initialize_paired_problem(
    engine: &mut NativePhasorThermodynamicEngine,
    seed: u64,
) -> Vec<Complex32> {
    let mut target = Vec::with_capacity(engine.node_count());
    for node in 0..engine.node_count() {
        let target_phase = 0.25;
        let target_value = Complex32::from_polar(1.0, target_phase);
        target.push(target_value);
        // Ambos métodos reciben exactamente la misma frontera parcial.
        engine.stimulus[node] = if splitmix64(seed.rotate_left(53) ^ node as u64) & 15 == 0 {
            target_value
        } else {
            Complex32::new(0.0, 0.0)
        };

        let corrupt = unit_from_u64(splitmix64(seed.rotate_left(17) ^ node as u64)) < 0.30;
        let jitter =
            0.35 * (2.0 * unit_from_u64(splitmix64(seed.rotate_left(41) ^ node as u64)) - 1.0);
        let initial_phase =
            target_phase + if corrupt { std::f32::consts::PI } else { 0.0 } + jitter;
        let amplitude = 0.65 + 0.70 * unit_from_u64(splitmix64(seed.rotate_left(7) ^ node as u64));
        engine.phasors[node] = Complex32::from_polar(amplitude, initial_phase);
    }
    target
}

fn target_alignment(state: &[Complex32], target: &[Complex32]) -> f32 {
    state
        .iter()
        .zip(target)
        .map(|(state, target)| {
            let magnitude = state.norm() * target.norm();
            if magnitude <= f32::EPSILON {
                0.0
            } else {
                ((state.conj() * *target).re / magnitude + 1.0) * 0.5
            }
        })
        .sum::<f32>()
        / state.len().max(1) as f32
}
