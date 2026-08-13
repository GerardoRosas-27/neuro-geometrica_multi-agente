//! Benchmark de la arquitectura híbrida Softmax + Fasor-RFF + motor termodinámico CDT.
//!
//! Compara tres modos sobre secuencias sintéticas:
//! - `softmax_only`: atención digital pura (baseline O(N²)).
//! - `rff_reservoir`: reservorio Langevin lineal O(N·D).
//! - `hybrid_full`: bucle cerrado con frontera CTP y plasticidad.

use cdt_rqm_epr::hybrid_thermo_attention::{
    digital_softmax_attention, HybridThermoAttention, HybridThermoAttentionConfig,
    LangevinReservoir, LangevinReservoirConfig, PhasorRffConfig, PhasorRffMap,
};
use cdt_rqm_epr::native_hybrid_phasor_cdt_engine::NativeHybridConfig;
use cdt_rqm_epr::native_phasor_thermodynamic_engine::NativePhasorMinimizerConfig;
use cdt_rqm_epr::native_rng::signed_unit;
use std::time::{Duration, Instant};

const DEFAULT_TRIALS: usize = 5;
const DEFAULT_SEQ_LEN: usize = 32;
const DEFAULT_D_MODEL: usize = 32;
const DEFAULT_D_V: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    elapsed: Duration,
    softmax_entropy: f64,
    ctp_bias_norm: f64,
    thermo_free_energy: f64,
    thermo_coherence: f64,
    plasticity_delta: f64,
    wake_pass_rate: f64,
}

impl Metrics {
    fn record(&mut self, elapsed: Duration, report: &cdt_rqm_epr::hybrid_thermo_attention::HybridThermoAttentionReport) {
        self.elapsed += elapsed;
        self.softmax_entropy += f64::from(report.softmax_entropy);
        self.ctp_bias_norm += f64::from(report.ctp_bias_norm);
        self.thermo_free_energy += f64::from(report.thermo_free_energy);
        self.thermo_coherence += f64::from(report.thermo_coherence);
        self.plasticity_delta += f64::from(report.plasticity_delta_norm);
        self.wake_pass_rate += f64::from(report.wake_gate_passed as u8);
    }

    fn mean(self, trials: usize) -> Self {
        let t = trials.max(1) as f64;
        Self {
            elapsed: self.elapsed.div_f64(t),
            softmax_entropy: self.softmax_entropy / t,
            ctp_bias_norm: self.ctp_bias_norm / t,
            thermo_free_energy: self.thermo_free_energy / t,
            thermo_coherence: self.thermo_coherence / t,
            plasticity_delta: self.plasticity_delta / t,
            wake_pass_rate: self.wake_pass_rate / t,
        }
    }

    fn print(self, seq_len: usize, d_model: usize, method: &str) {
        println!(
            "{seq_len},{d_model},{method},{:.3},{:.4},{:.4},{:.4},{:.4},{:.6},{:.2}",
            self.elapsed.as_secs_f64() * 1_000.0,
            self.softmax_entropy,
            self.ctp_bias_norm,
            self.thermo_free_energy,
            self.thermo_coherence,
            self.plasticity_delta,
            self.wake_pass_rate,
        );
    }
}

fn synthetic_sequence(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..d)
                .map(|j| signed_unit(seed ^ ((i as u64) << 16) ^ j as u64))
                .collect()
        })
        .collect()
}

fn bench_softmax_only(seq_len: usize, d_model: usize, d_v: usize, trials: usize) -> Metrics {
    let mut metrics = Metrics::default();
    for trial in 0..trials {
        let seed = 0x534F_4654 ^ trial as u64;
        let q = synthetic_sequence(seq_len, d_model, seed);
        let k = synthetic_sequence(seq_len, d_model, seed ^ 1);
        let v = synthetic_sequence(seq_len, d_v, seed ^ 2);
        let start = Instant::now();
        let (_, attention) = digital_softmax_attention(&q, &k, &v, None, 0.0);
        let elapsed = start.elapsed();
        let entropy: f32 = attention
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&p| p > 1.0e-7)
            .map(|&p| -p * p.ln())
            .sum::<f32>()
            / seq_len as f32;
        metrics.elapsed += elapsed;
        metrics.softmax_entropy += f64::from(entropy);
        metrics.wake_pass_rate += 1.0;
    }
    metrics.mean(trials)
}

fn bench_rff_reservoir(seq_len: usize, d_model: usize, d_v: usize, trials: usize) -> Metrics {
    let features = 64;
    let mut metrics = Metrics::default();
    for trial in 0..trials {
        let seed = 0x5246_46 ^ trial as u64;
        let q = synthetic_sequence(seq_len, d_model, seed);
        let k = synthetic_sequence(seq_len, d_model, seed ^ 1);
        let v = synthetic_sequence(seq_len, d_v, seed ^ 2);
        let rff = PhasorRffMap::new(d_model, PhasorRffConfig {
            features,
            ..Default::default()
        });
        let mut reservoir = LangevinReservoir::new(d_v, features, LangevinReservoirConfig::default());
        let start = Instant::now();
        for j in 0..seq_len {
            let (phi_r, phi_i) = rff.project(&k[j]);
            reservoir.inject(&phi_r, &phi_i, &v[j]);
        }
        for i in 0..seq_len {
            let (phi_r, phi_i) = rff.project(&q[i]);
            let _ = reservoir.query(&phi_r, &phi_i);
        }
        let elapsed = start.elapsed();
        metrics.elapsed += elapsed;
        metrics.softmax_entropy += 0.0;
        metrics.wake_pass_rate += 1.0;
    }
    metrics.mean(trials)
}

fn bench_hybrid_full(seq_len: usize, d_model: usize, d_v: usize, trials: usize) -> Metrics {
    let mut metrics = Metrics::default();
    for trial in 0..trials {
        let config = HybridThermoAttentionConfig {
            d_model,
            d_v,
            rff: PhasorRffConfig {
                features: 64,
                ..Default::default()
            },
            cdt_nodes: 128,
            cdt_spatial_degree: 4,
            hybrid: NativeHybridConfig {
                minimizer: NativePhasorMinimizerConfig {
                    max_iterations: 60,
                    residual_tolerance: 5.0e-3,
                    ..Default::default()
                },
                minimum_relative_energy_drop: 0.0,
                maximum_residual: 5.0e-2,
                minimum_magnetic_coherence: 0.6,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine = HybridThermoAttention::new(config).expect("engine");
        let seed = 0x4859_4252 ^ trial as u64;
        let tokens = synthetic_sequence(seq_len, d_model, seed);
        let values = synthetic_sequence(seq_len, d_v, seed ^ 1);
        let start = Instant::now();
        let (_, report) = engine.forward(&tokens, &values).expect("forward");
        let elapsed = start.elapsed();
        metrics.record(elapsed, &report);
    }
    metrics.mean(trials)
}

fn main() {
    let trials = env_usize("HYBRID_THERMO_TRIALS", DEFAULT_TRIALS);
    let seq_len = env_usize("HYBRID_THERMO_SEQ_LEN", DEFAULT_SEQ_LEN).max(4);
    let d_model = env_usize("HYBRID_THERMO_D_MODEL", DEFAULT_D_MODEL).max(4);
    let d_v = env_usize("HYBRID_THERMO_D_V", DEFAULT_D_V).max(2);

    println!("benchmark=hybrid_thermo_attention_softmax_rff_cdt");
    println!(
        "config,trials={trials},seq_len={seq_len},d_model={d_model},d_v={d_v}"
    );
    println!(
        "seq_len,d_model,method,mean_ms,softmax_entropy,ctp_bias_norm,\
         thermo_F,thermo_coherence,plasticity_delta,wake_pass_rate"
    );

    bench_softmax_only(seq_len, d_model, d_v, trials)
        .print(seq_len, d_model, "softmax_only");
    bench_rff_reservoir(seq_len, d_model, d_v, trials)
        .print(seq_len, d_model, "rff_reservoir");
    bench_hybrid_full(seq_len, d_model, d_v, trials)
        .print(seq_len, d_model, "hybrid_full");
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
