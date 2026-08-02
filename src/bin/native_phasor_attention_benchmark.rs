use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorInferencePolicy, NativePhasorMinimizerConfig,
    NativePhasorThermodynamicEngine,
};
use cdt_rqm_epr::native_rng::{splitmix64, unit_from_u64};
use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use num_complex::Complex32;
use std::time::{Duration, Instant};

const TRIALS: usize = 8;
const ITERATIONS: usize = 24;

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    elapsed: Duration,
    free_energy_per_node: f64,
    residual: f64,
    coherence: f64,
    evaluations: usize,
    accepted: usize,
    attention_entropy: f64,
    peak_attention_gain: f64,
}

impl Metrics {
    fn record(
        &mut self,
        engine: &NativePhasorThermodynamicEngine,
        elapsed: Duration,
        result: &cdt_rqm_epr::native_phasor_thermodynamic_engine::NativePhasorMinimizationReport,
    ) {
        let report = engine.report();
        self.elapsed += elapsed;
        self.free_energy_per_node += f64::from(report.free_energy) / report.nodes.max(1) as f64;
        self.residual += f64::from(report.gradient_residual);
        self.coherence += f64::from(report.phase_coherence);
        self.evaluations += result.energy_evaluations;
        self.accepted += result.accepted_steps;
        self.attention_entropy += f64::from(result.mean_attention_entropy);
        self.peak_attention_gain += f64::from(result.peak_attention_gain);
    }

    fn mean(self) -> Self {
        Self {
            elapsed: self.elapsed.div_f64(TRIALS as f64),
            free_energy_per_node: self.free_energy_per_node / TRIALS as f64,
            residual: self.residual / TRIALS as f64,
            coherence: self.coherence / TRIALS as f64,
            evaluations: self.evaluations / TRIALS,
            accepted: self.accepted / TRIALS,
            attention_entropy: self.attention_entropy / TRIALS as f64,
            peak_attention_gain: self.peak_attention_gain / TRIALS as f64,
        }
    }

    fn print(self, nodes: usize, method: &str) {
        let mean = self.mean();
        println!(
            "{nodes},{method},{:.3},{:.7},{:.3e},{:.6},{},{},{:.6},{:.3}",
            mean.elapsed.as_secs_f64() * 1_000.0,
            mean.free_energy_per_node,
            mean.residual,
            mean.coherence,
            mean.evaluations,
            mean.accepted,
            mean.attention_entropy,
            mean.peak_attention_gain,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("benchmark=inferencia_fasorial_energia_vs_atencion_residual");
    println!("objetivo=misma_energia_libre aceptacion=armijo_exacta");
    println!("presupuesto={ITERATIONS}_iteraciones ensayos={TRIALS}");
    println!(
        "nodes,method,mean_ms,F_per_node,residual,coherence,evaluations,accepted,\
         attention_entropy,peak_attention_gain"
    );

    let mut energy_wins = 0usize;
    let mut residual_wins = 0usize;
    let mut comparisons = 0usize;
    let mut total_energy_improvement = 0.0f64;
    let mut total_residual_improvement = 0.0f64;

    for nodes in [256usize, 1_024, 4_096] {
        let template = fixture(nodes)?;
        let mut baseline_metrics = Metrics::default();
        let mut hybrid_metrics = Metrics::default();

        for trial in 0..TRIALS {
            let seed = 0xA77E_2026 ^ nodes as u64 ^ (trial as u64).rotate_left(17);
            let mut initial = template.clone();
            randomize_problem(&mut initial, seed);

            let mut baseline = initial.clone();
            let started = Instant::now();
            let baseline_result = baseline.minimize_free_energy(minimizer_config(0.0));
            baseline_metrics.record(&baseline, started.elapsed(), &baseline_result);

            let mut hybrid = initial;
            let started = Instant::now();
            let hybrid_result = hybrid.minimize_free_energy(minimizer_config(0.65));
            hybrid_metrics.record(&hybrid, started.elapsed(), &hybrid_result);

            let baseline_report = baseline.report();
            let hybrid_report = hybrid.report();
            if hybrid_report.free_energy < baseline_report.free_energy {
                energy_wins += 1;
            }
            if hybrid_report.gradient_residual < baseline_report.gradient_residual {
                residual_wins += 1;
            }
            total_energy_improvement +=
                f64::from(baseline_report.free_energy - hybrid_report.free_energy)
                    / f64::from(baseline_report.free_energy.abs()).max(1.0);
            total_residual_improvement +=
                f64::from(baseline_report.gradient_residual - hybrid_report.gradient_residual)
                    / f64::from(baseline_report.gradient_residual).max(1.0e-9);
            comparisons += 1;

            if hybrid_report.free_energy > hybrid_result.initial.free_energy + 1.0e-4 {
                return Err("la rama híbrida aumentó la energía libre".into());
            }
        }

        baseline_metrics.print(nodes, "energia_armijo");
        hybrid_metrics.print(nodes, "energia_atencion_fasorial");
    }

    let energy_improvement = 100.0 * total_energy_improvement / comparisons as f64;
    let residual_improvement = 100.0 * total_residual_improvement / comparisons as f64;
    println!(
        "resultado,comparaciones={comparisons},energy_wins={energy_wins},\
         residual_wins={residual_wins},mean_relative_energy_improvement={energy_improvement:.4}%,\
         mean_relative_residual_improvement={residual_improvement:.4}%"
    );
    let improved = energy_wins * 2 > comparisons
        && residual_wins * 2 > comparisons
        && energy_improvement > 0.0
        && residual_improvement > 0.0;
    println!(
        "veredicto={}",
        if improved {
            "hibrido_mejora_con_presupuesto_fijo"
        } else {
            "sin_mejora_consistente"
        }
    );
    Ok(())
}

fn minimizer_config(attention_strength: f32) -> NativePhasorMinimizerConfig {
    NativePhasorMinimizerConfig {
        max_iterations: ITERATIONS,
        energy_tolerance: 0.0,
        residual_tolerance: 0.0,
        topological_warm_start: false,
        attention_strength,
        attention_temperature: 0.75,
        attention_max_gain: 4.0,
        // Este banco aísla la atención pura: la programación adaptativa la
        // apagaría a mitad del descenso y mediría otra cosa.
        inference_policy: NativePhasorInferencePolicy::Fixed,
        ..NativePhasorMinimizerConfig::default()
    }
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
            radial_strength: 3.0,
            target_amplitude: 1.0,
            confinement: 0.02,
            stimulus_gain: 0.8,
            entropy_weight: 0.15,
            temperature_scale: 0.10,
            noise_scale: 0.0,
            max_amplitude: 2.0,
            ..NativePhasorConfig::default()
        },
    )?)
}

fn randomize_problem(engine: &mut NativePhasorThermodynamicEngine, seed: u64) {
    for node in 0..engine.node_count() {
        let phase = std::f32::consts::TAU * unit_from_u64(splitmix64(seed ^ node as u64));
        let amplitude = 0.55 + 0.90 * unit_from_u64(splitmix64(seed.rotate_left(19) ^ node as u64));
        engine.phasors[node] = Complex32::from_polar(amplitude, phase);

        let selector = unit_from_u64(splitmix64(seed.rotate_left(37) ^ node as u64));
        engine.stimulus[node] = if selector < 0.125 {
            let target_phase = if selector < 0.0625 {
                0.35
            } else {
                std::f32::consts::PI + 0.35
            };
            Complex32::from_polar(1.0, target_phase)
        } else {
            Complex32::new(0.0, 0.0)
        };
    }
}
