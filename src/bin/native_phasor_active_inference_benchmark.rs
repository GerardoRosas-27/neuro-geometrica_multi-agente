use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    NativePhasorActiveInferenceConfig, NativePhasorConfig, NativePhasorMinimizerConfig,
    NativePhasorThermodynamicEngine,
};
use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use num_complex::Complex32;
use std::time::{Duration, Instant};

const TRIALS: usize = 5;
const PASSES: usize = 100;

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    elapsed: Duration,
    free_energy_per_node: f64,
    residual: f64,
    coherence: f64,
    operations: usize,
    accepted: usize,
    entropy_error: f64,
    sampled_energy_relative_error: f64,
}

impl Metrics {
    fn mean_ms(self) -> f64 {
        self.elapsed.as_secs_f64() * 1_000.0 / TRIALS as f64
    }

    fn mean_free_energy_per_node(self) -> f64 {
        self.free_energy_per_node / TRIALS as f64
    }

    fn record(
        &mut self,
        engine: &NativePhasorThermodynamicEngine,
        elapsed: Duration,
        operations: usize,
        accepted: usize,
        sampled_entropy: Option<f32>,
        sampled_internal_energy: Option<f32>,
    ) {
        let report = engine.report();
        self.elapsed += elapsed;
        self.free_energy_per_node += report.free_energy as f64 / report.nodes.max(1) as f64;
        self.residual += report.gradient_residual as f64;
        self.coherence += report.phase_coherence as f64;
        self.operations += operations;
        self.accepted += accepted;
        if let Some(estimate) = sampled_entropy {
            self.entropy_error += (estimate - report.entropy).abs() as f64;
        }
        if let Some(estimate) = sampled_internal_energy {
            self.sampled_energy_relative_error += (estimate - report.internal_energy).abs() as f64
                / report.internal_energy.abs().max(1.0) as f64;
        }
    }

    fn print(self, method: &str) {
        println!(
            "{method},{:.3},{:.6},{:.3e},{:.6},{:.1},{:.1},{:.3e},{:.3e}",
            self.mean_ms(),
            self.mean_free_energy_per_node(),
            self.residual / TRIALS as f64,
            self.coherence / TRIALS as f64,
            self.operations as f64 / TRIALS as f64,
            self.accepted as f64 / TRIALS as f64,
            self.entropy_error / TRIALS as f64,
            self.sampled_energy_relative_error / TRIALS as f64,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("benchmark=motor_fasorial_nativo_aislado");
    println!("dependencias_excluidas=Gemma,Transformer,RQM,EPR,cognición,sueño");
    println!("topologia=grafo_CDT_inmutable pachner=omitido_sin_complejo_simplicial");
    println!("presupuesto={} barridos/iteraciones por ensayo", PASSES);
    println!(
        "nodes,method,mean_ms,F_per_node,residual,coherence,operations,accepted,\
         entropy_mc_abs_error,energy_mc_relative_error"
    );

    let mut armijo_wins = 0usize;
    for nodes in [128usize, 512, 2_048] {
        let template = fixture(nodes)?;
        let mut gradient_metrics = Metrics::default();
        let mut gibbs_metrics = Metrics::default();
        let mut active_metrics = Metrics::default();

        for trial in 0..TRIALS {
            let mut initial = template.clone();
            randomize_state(&mut initial, 0xB3AC_2026 ^ nodes as u64 ^ trial as u64);

            let mut gradient = initial.clone();
            let started = Instant::now();
            let report = gradient.minimize_free_energy(NativePhasorMinimizerConfig {
                max_iterations: PASSES,
                topological_warm_start: false,
                ..NativePhasorMinimizerConfig::default()
            });
            gradient_metrics.record(
                &gradient,
                started.elapsed(),
                report.energy_evaluations,
                report.accepted_steps,
                None,
                None,
            );

            let mut gibbs = initial.clone();
            let started = Instant::now();
            let report = gibbs.active_inference(NativePhasorActiveInferenceConfig {
                sweeps: PASSES,
                burn_in_sweeps: PASSES / 5,
                local_learning_rate: 0.0,
                entropy_samples: 1_024,
                seed: 0x61BB_5200 ^ trial as u64,
                ..NativePhasorActiveInferenceConfig::default()
            });
            gibbs_metrics.record(
                &gibbs,
                started.elapsed(),
                report.gibbs_proposals,
                report.gibbs_accepted,
                Some(report.sampled_entropy),
                Some(report.sampled_mean_internal_energy),
            );

            let mut active = initial;
            let started = Instant::now();
            let report = active.active_inference(NativePhasorActiveInferenceConfig {
                sweeps: PASSES,
                burn_in_sweeps: PASSES / 5,
                local_learning_rate: 0.35,
                entropy_samples: 1_024,
                seed: 0xAC71_1EFE ^ trial as u64,
                ..NativePhasorActiveInferenceConfig::default()
            });
            active_metrics.record(
                &active,
                started.elapsed(),
                report.gibbs_proposals,
                report.gibbs_accepted + report.local_updates_accepted,
                Some(report.sampled_entropy),
                Some(report.sampled_mean_internal_energy),
            );
        }

        print!("{nodes},");
        gradient_metrics.print("gradiente_global_armijo");
        print!("{nodes},");
        gibbs_metrics.print("gibbs_metropolis_local");
        print!("{nodes},");
        active_metrics.print("active_gibbs_gradiente_local");
        if gradient_metrics.mean_ms() < gibbs_metrics.mean_ms()
            && gradient_metrics.mean_ms() < active_metrics.mean_ms()
            && gradient_metrics.mean_free_energy_per_node()
                <= gibbs_metrics.mean_free_energy_per_node()
            && gradient_metrics.mean_free_energy_per_node()
                <= active_metrics.mean_free_energy_per_node()
        {
            armijo_wins += 1;
        }
    }
    if armijo_wins != 3 {
        return Err(format!(
            "regresión: Armijo ganó tiempo y energía sólo en {armijo_wins}/3 escalas"
        )
        .into());
    }
    println!("resultado=gradiente_global_armijo solver_produccion=confirmado escalas_ganadas=3/3");
    Ok(())
}

fn fixture(nodes: usize) -> Result<NativePhasorThermodynamicEngine, Box<dyn std::error::Error>> {
    let core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: 1,
        nodes_per_slice: nodes,
        spatial_degree: 3,
        temporal_degree: 1,
        temperature: 1.0,
        seed: 0xF17E_0000 ^ nodes as u64,
        ..NativeThermoCdtConfig::default()
    });
    Ok(NativePhasorThermodynamicEngine::from_core(
        &core,
        NativePhasorConfig {
            coupling_strength: 1.0,
            radial_strength: 4.0,
            target_amplitude: 1.0,
            confinement: 0.02,
            stimulus_gain: 0.0,
            entropy_weight: 0.15,
            temperature_scale: 0.10,
            noise_scale: 0.0,
            max_amplitude: 2.0,
            ..NativePhasorConfig::default()
        },
    )?)
}

fn randomize_state(engine: &mut NativePhasorThermodynamicEngine, seed: u64) {
    for (node, phasor) in engine.phasors.iter_mut().enumerate() {
        let phase = std::f32::consts::TAU * unit_from_u64(splitmix64(seed ^ node as u64));
        let amplitude = 0.75 + 0.50 * unit_from_u64(splitmix64(seed.rotate_left(19) ^ node as u64));
        *phasor = Complex32::from_polar(amplitude, phase);
    }
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
