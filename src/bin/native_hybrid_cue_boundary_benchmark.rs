//! A/B pareado del ciclo wake/sleep/recuerdo según cómo entre la cue.
//!
//! Variante 1: la cue es sólo condición inicial (comportamiento actual).
//! Variante 2: la cue entra en F como frontera `-g·Re(ψ*·s)`.
//! Variante 3: la misma frontera más el ciclo Handshake+atención adaptativo.
//!
//! F no es comparable entre la variante 1 y las demás porque el funcional
//! cambia al añadir el término de frontera. Lo comparable en las tres es la
//! alineación con el patrón objetivo, si el gate deja consolidar y el coste.

use cdt_rqm_epr::native_hybrid_phasor_cdt_engine::{
    NativeHybridConfig, NativeHybridPhasorCdtEngine, NativePhasorCue,
};
use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorInferencePolicy, NativePhasorMinimizerConfig,
};
use cdt_rqm_epr::native_rng::{splitmix64, unit_from_u64};
use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use num_complex::Complex32;
use std::time::{Duration, Instant};

const TRIALS: usize = 8;
const TARGET_PHASE: f32 = 0.25;

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    wake_elapsed: Duration,
    sleep_elapsed: Duration,
    wake_alignment: f64,
    recall_alignment: f64,
    residual: f64,
    coherence: f64,
    gate_passed: usize,
    accepted: usize,
    storage_delta: f64,
    trials: usize,
}

impl Metrics {
    fn print(self, nodes: usize, method: &str) {
        let trials = self.trials.max(1) as f64;
        println!(
            "{nodes},{method},{:.3},{:.3},{:.6},{:.6},{:.3e},{:.6},{},{},{:.6}",
            self.wake_elapsed.as_secs_f64() * 1_000.0 / trials,
            self.sleep_elapsed.as_secs_f64() * 1_000.0 / trials,
            self.wake_alignment / trials,
            self.recall_alignment / trials,
            self.residual / trials,
            self.coherence / trials,
            self.gate_passed,
            self.accepted,
            self.storage_delta / trials,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("benchmark=cue_como_condicion_inicial_vs_cue_como_frontera");
    println!("control=mismo_core_estado_corrompido_cue_semilla_presupuesto");
    println!("nota=F_no_comparable_entre_variante_1_y_el_resto");
    println!("trials={TRIALS}");
    println!(
        "nodes,method,wake_ms,sleep_ms,wake_alignment,recall_alignment,residual,coherence,\
         gate_passed,accepted,storage_delta"
    );

    for nodes in [256usize, 1_024] {
        let mut metrics = [Metrics::default(); 3];

        for trial in 0..TRIALS {
            let seed = 0x0CDE_B00D_u64 ^ nodes as u64 ^ (trial as u64).rotate_left(31);
            let target = Complex32::from_polar(1.0, TARGET_PHASE);
            let cue = partial_cue(nodes, seed);
            let recall_cue = partial_cue(nodes, seed.rotate_left(11));

            for (index, (_, boundary, policy)) in VARIANTS.iter().enumerate() {
                let mut engine = engine(nodes, seed, *boundary, *policy)?;

                corrupt_state(&mut engine, seed);
                let started = Instant::now();
                let wake = engine.infer_and_stage(&cue)?;
                metrics[index].wake_elapsed += started.elapsed();
                metrics[index].wake_alignment +=
                    f64::from(target_alignment(&engine.phasor.phasors, target));
                metrics[index].residual +=
                    f64::from(wake.minimization.final_report.gradient_residual);
                metrics[index].coherence +=
                    f64::from(wake.minimization.final_report.phase_coherence);
                metrics[index].gate_passed += usize::from(wake.gate.passed);

                let started = Instant::now();
                let sleep = engine.sleep_consolidate()?;
                metrics[index].sleep_elapsed += started.elapsed();
                metrics[index].accepted += sleep.accepted;
                metrics[index].storage_delta += f64::from(sleep.mean_storage_delta_free_energy);

                corrupt_state(&mut engine, seed.rotate_left(23));
                engine.infer_and_stage(&recall_cue)?;
                metrics[index].recall_alignment +=
                    f64::from(target_alignment(&engine.phasor.phasors, target));
                metrics[index].trials += 1;
            }
        }

        for (index, (name, ..)) in VARIANTS.iter().enumerate() {
            metrics[index].print(nodes, name);
        }
    }
    Ok(())
}

const VARIANTS: [(&str, bool, NativePhasorInferencePolicy); 3] = [
    (
        "cue_inicial_armijo",
        false,
        NativePhasorInferencePolicy::Fixed,
    ),
    (
        "cue_frontera_armijo",
        true,
        NativePhasorInferencePolicy::Fixed,
    ),
    (
        "cue_frontera_hibrido",
        true,
        NativePhasorInferencePolicy::Adaptive,
    ),
];

fn engine(
    nodes: usize,
    seed: u64,
    cue_as_boundary: bool,
    policy: NativePhasorInferencePolicy,
) -> Result<NativeHybridPhasorCdtEngine, Box<dyn std::error::Error>> {
    let mut core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: 1,
        nodes_per_slice: nodes,
        spatial_degree: 4,
        temporal_degree: 1,
        temperature: 0.0,
        diffusion: 0.0,
        pilot_gain: 0.0,
        amplitude_decay: 0.0,
        seed,
        ..NativeThermoCdtConfig::default()
    });
    // Transporte de fase nulo: el acoplamiento prefiere fases iguales y deja
    // la fase global como gauge libre. Sólo la frontera puede fijarla.
    core.edge_phase.fill(0.0);
    let modulated = cue_as_boundary && policy == NativePhasorInferencePolicy::Adaptive;
    Ok(NativeHybridPhasorCdtEngine::from_core(
        core,
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
        NativeHybridConfig {
            minimizer: NativePhasorMinimizerConfig {
                max_iterations: 64,
                residual_tolerance: 5.0e-3,
                topological_warm_start: false,
                handshake_strength: if modulated { 0.65 } else { 0.0 },
                attention_strength: if modulated { 0.55 } else { 0.0 },
                attention_temperature: 0.75,
                attention_max_gain: 3.0,
                attention_ignition_threshold: 0.001,
                handshake_max_gain: 3.0,
                inference_policy: policy,
                ..NativePhasorMinimizerConfig::default()
            },
            minimum_magnetic_coherence: 0.80,
            cue_as_boundary,
            ..NativeHybridConfig::default()
        },
    )?)
}

fn partial_cue(nodes: usize, seed: u64) -> Vec<NativePhasorCue> {
    (0..nodes)
        .filter(|node| splitmix64(seed.rotate_left(53) ^ *node as u64) & 7 == 0)
        .map(|node| NativePhasorCue {
            node,
            amplitude: 1.0,
            phase: TARGET_PHASE,
        })
        .collect()
}

fn corrupt_state(engine: &mut NativeHybridPhasorCdtEngine, seed: u64) {
    for (node, phasor) in engine.phasor.phasors.iter_mut().enumerate() {
        let flipped = unit_from_u64(splitmix64(seed.rotate_left(17) ^ node as u64)) < 0.30;
        let jitter =
            0.35 * (2.0 * unit_from_u64(splitmix64(seed.rotate_left(41) ^ node as u64)) - 1.0);
        let phase = TARGET_PHASE + if flipped { std::f32::consts::PI } else { 0.0 } + jitter;
        let amplitude = 0.65 + 0.70 * unit_from_u64(splitmix64(seed.rotate_left(7) ^ node as u64));
        *phasor = Complex32::from_polar(amplitude, phase);
    }
}

fn target_alignment(state: &[Complex32], target: Complex32) -> f32 {
    state
        .iter()
        .map(|value| {
            let magnitude = value.norm() * target.norm();
            if magnitude <= f32::EPSILON {
                0.0
            } else {
                ((value.conj() * target).re / magnitude + 1.0) * 0.5
            }
        })
        .sum::<f32>()
        / state.len().max(1) as f32
}
