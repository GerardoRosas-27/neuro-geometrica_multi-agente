//! Arquitectura híbrida: Softmax digital + motor termodinámico fasorial-CDT.
//!
//! Combina tres capas en un bucle cerrado:
//! 1. **Digital (Softmax)**: calcula el apretón de manos exacto `A = Softmax(QK^T/√d + α·B_CTP)`.
//! 2. **Reservorio Langevin (Fasor-RFF)**: acumula `S = Σ φ(K)†⊗V` y `z = Σ φ(K)†` en O(N).
//! 3. **Motor híbrido CDT**: inyecta `A` como frontera CTP, relaja por energía libre y
//!    cristaliza atractores que retroalimentan `B_CTP` y las proyecciones `W_Q`, `W_K`.
//!
//! ```text
//!   Q,K ──► Softmax(QK^T + α·B_CTP) ──► A_ij ──► frontera CTP ──► motor híbrido
//!     │                                        ▲                      │
//!     └──► φ_RFF ──► reservorio Langevin ──────┘                      ▼
//!                                                          B_CTP, ΔW_Q, ΔW_K
//! ```

use crate::native_hybrid_phasor_cdt_engine::{
    HybridEngineLearnedState, NativeHybridConfig, NativeHybridPhasorCdtEngine, NativePhasorCue,
};
use crate::native_phasor_thermodynamic_engine::NativePhasorConfig;
use crate::native_rng::{gaussian_from_counter, splitmix64, signed_unit};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use std::fmt;
use serde::{Deserialize, Serialize};

const EPSILON: f32 = 1.0e-7;

// ── Configuración ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct PhasorRffConfig {
    /// Dimensión del espacio de fasores (frecuencias muestreadas).
    pub features: usize,
    /// Desviación estándar de ω ~ N(0, σ²I) según Bochner.
    pub sigma: f32,
    pub seed: u64,
}

impl Default for PhasorRffConfig {
    fn default() -> Self {
        Self {
            features: 64,
            sigma: 1.0,
            seed: 0x5246_4620_5048_4153,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LangevinReservoirConfig {
    /// Tasa de disipación γ: ventana de memoria exponencial.
    pub gamma: f32,
    /// Paso de integración dt.
    pub dt: f32,
    /// Temperatura efectiva k_B T para fluctuaciones.
    pub kbt: f32,
    /// Pasos de Langevin por token inyectado.
    pub steps_per_token: usize,
    pub seed: u64,
}

impl Default for LangevinReservoirConfig {
    fn default() -> Self {
        Self {
            gamma: 0.5,
            dt: 0.05,
            kbt: 0.01,
            steps_per_token: 1,
            seed: 0x4C41_4E47_4556_494E,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HybridThermoAttentionConfig {
    pub d_model: usize,
    pub d_v: usize,
    pub rff: PhasorRffConfig,
    pub reservoir: LangevinReservoirConfig,
    /// Acoplamiento α entre sesgo CTP y logits de Softmax.
    pub ctp_coupling_alpha: f32,
    /// Mezcla entre salida Softmax y salida del reservorio Langevin [0, 1].
    pub thermo_blend: f32,
    /// Top-k acoplamientos de A_ij que se inyectan como frontera CTP.
    pub boundary_top_k: usize,
    /// Umbral mínimo de A_ij para inyectar frontera.
    pub boundary_threshold: f32,
    /// Tasa de plasticidad η para W_Q, W_K.
    pub plasticity_eta: f32,
    /// Temperatura del ruido en la regla de plasticidad Langevin-Hebbiana.
    pub plasticity_temperature: f32,
    /// Nodos del grafo CDT/fasorial.
    pub cdt_nodes: usize,
    pub cdt_spatial_degree: usize,
    pub cdt_seed: u64,
    pub hybrid: NativeHybridConfig,
    pub phasor: NativePhasorConfig,
}

impl Default for HybridThermoAttentionConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            d_v: 16,
            rff: PhasorRffConfig::default(),
            reservoir: LangevinReservoirConfig::default(),
            ctp_coupling_alpha: 0.35,
            thermo_blend: 0.4,
            boundary_top_k: 4,
            boundary_threshold: 0.05,
            plasticity_eta: 0.002,
            plasticity_temperature: 0.005,
            cdt_nodes: 256,
            cdt_spatial_degree: 4,
            cdt_seed: 0x4354_445F_4859_4252,
            hybrid: NativeHybridConfig::default(),
            phasor: NativePhasorConfig {
                temperature_scale: 0.02,
                noise_scale: 0.5,
                ..NativePhasorConfig::default()
            },
        }
    }
}

// ── Informes ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HybridThermoAttentionReport {
    pub tick: u64,
    pub sequence_length: usize,
    pub softmax_entropy: f32,
    pub reservoir_denominator: f32,
    pub ctp_bias_norm: f32,
    pub thermo_free_energy: f32,
    pub thermo_coherence: f32,
    pub plasticity_delta_norm: f32,
    pub wake_gate_passed: bool,
    pub pending_attractors: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HybridThermoAttentionError {
    DimensionMismatch { expected: usize, got: usize },
    EmptySequence,
    Hybrid(String),
}

impl fmt::Display for HybridThermoAttentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => write!(
                formatter,
                "dimensión esperada {expected}, recibida {got}"
            ),
            Self::EmptySequence => write!(formatter, "la secuencia está vacía"),
            Self::Hybrid(message) => write!(formatter, "error híbrido: {message}"),
        }
    }
}

impl std::error::Error for HybridThermoAttentionError {}

// ── Mapeo RFF al espacio de fasores ─────────────────────────────────────────

/// Proyector Random Fourier Features: ϕ(x) = (1/√D) · exp(i ω^T x).
#[derive(Clone, Debug)]
pub struct PhasorRffMap {
    omega: Vec<Vec<f32>>,
    d_model: usize,
    features: usize,
    scale: f32,
}

impl PhasorRffMap {
    pub fn new(d_model: usize, config: PhasorRffConfig) -> Self {
        let features = config.features.max(1);
        let scale = (features as f32).sqrt().recip();
        let mut omega = Vec::with_capacity(features);
        for m in 0..features {
            let mut row = Vec::with_capacity(d_model);
            for d in 0..d_model {
                let seed = config.seed ^ ((m as u64) << 32) ^ (d as u64);
                row.push(signed_unit(seed) * config.sigma);
            }
            omega.push(row);
        }
        Self {
            omega,
            d_model,
            features,
            scale,
        }
    }

    pub fn features(&self) -> usize {
        self.features
    }

    /// Proyecta x ∈ R^d a (real, imag) ∈ R^D × R^D.
    pub fn project(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(x.len(), self.d_model);
        let mut real = vec![0.0; self.features];
        let mut imag = vec![0.0; self.features];
        for (m, omega_m) in self.omega.iter().enumerate() {
            let mut dot = 0.0f32;
            for (d, &xi) in x.iter().enumerate() {
                dot += omega_m[d] * xi;
            }
            real[m] = self.scale * dot.cos();
            imag[m] = self.scale * dot.sin();
        }
        (real, imag)
    }

    /// Producto interno aproximado k(x,y) ≈ Re{ϕ(x)† · ϕ(y)}.
    pub fn kernel_approx(&self, x: &[f32], y: &[f32]) -> f32 {
        let (xr, xi) = self.project(x);
        let (yr, yi) = self.project(y);
        dot_complex(&xr, &xi, &yr, &yi)
    }
}

// ── Reservorio termodinámico Langevin ───────────────────────────────────────

/// Tensor de estado S ∈ C^{D×d_v} y vector z ∈ C^D con dinámica de Langevin.
#[derive(Clone, Debug)]
pub struct LangevinReservoir {
    s_real: Vec<Vec<f32>>,
    s_imag: Vec<Vec<f32>>,
    z_real: Vec<f32>,
    z_imag: Vec<f32>,
    d_v: usize,
    config: LangevinReservoirConfig,
    tick: u64,
}

impl LangevinReservoir {
    pub fn new(d_v: usize, features: usize, config: LangevinReservoirConfig) -> Self {
        Self {
            s_real: vec![vec![0.0; d_v]; features],
            s_imag: vec![vec![0.0; d_v]; features],
            z_real: vec![0.0; features],
            z_imag: vec![0.0; features],
            d_v,
            config,
            tick: 0,
        }
    }

    pub fn reset(&mut self) {
        for row in &mut self.s_real {
            row.fill(0.0);
        }
        for row in &mut self.s_imag {
            row.fill(0.0);
        }
        self.z_real.fill(0.0);
        self.z_imag.fill(0.0);
        self.tick = 0;
    }

    /// Inyecta un par (φ(K), V) y avanza la dinámica de Langevin.
    pub fn inject(&mut self, phi_real: &[f32], phi_imag: &[f32], value: &[f32]) {
        debug_assert_eq!(phi_real.len(), self.z_real.len());
        debug_assert_eq!(value.len(), self.d_v);
        let gamma = self.config.gamma;
        let dt = self.config.dt;
        let kbt = self.config.kbt;
        let noise_scale = (2.0 * gamma * kbt * dt).sqrt();
        let seed = self.config.seed ^ self.tick.rotate_left(17);

        for m in 0..self.z_real.len() {
            let phi_r = phi_real[m];
            let phi_i = phi_imag[m];

            // dz = (-γ z + φ*) dt
            self.z_real[m] += (-gamma * self.z_real[m] + phi_r) * dt;
            self.z_imag[m] += (-gamma * self.z_imag[m] - phi_i) * dt;

            for d in 0..self.d_v {
                let v = value[d];
                // dS = (-γ S + φ* ⊗ V) dt + ruido
                let noise = gaussian_from_counter(seed, (m as u64) * self.d_v as u64 + d as u64)
                    * noise_scale;
                self.s_real[m][d] +=
                    (-gamma * self.s_real[m][d] + phi_r * v) * dt + noise;
                self.s_imag[m][d] +=
                    (-gamma * self.s_imag[m][d] - phi_i * v) * dt;
            }
        }
        self.tick = self.tick.wrapping_add(1);
    }

    /// Proyección O(1) del apretón de manos: Out = Re{ϕ(Q)† S} / Re{ϕ(Q)† z}.
    pub fn query(&self, phi_real: &[f32], phi_imag: &[f32]) -> (Vec<f32>, f32) {
        let mut output = vec![0.0; self.d_v];
        let mut denom = 0.0f32;
        for m in 0..self.z_real.len() {
            let qr = phi_real[m];
            let qi = phi_imag[m];
            // Re{ϕ† z} = ϕ_r z_r + ϕ_i z_i  (con ϕ = e^{iθ}, ϕ† contribuye conjugado)
            denom += qr * self.z_real[m] + qi * self.z_imag[m];
            for d in 0..self.d_v {
                output[d] += qr * self.s_real[m][d] + qi * self.s_imag[m][d];
            }
        }
        let safe_denom = denom.abs().max(EPSILON);
        for v in &mut output {
            *v /= safe_denom;
        }
        (output, denom)
    }

    /// Distribución de equilibrio aproximada P_eq(j) ∝ |z_j| para plasticidad.
    pub fn equilibrium_weights(&self) -> Vec<f32> {
        let mut weights: Vec<f32> = self
            .z_real
            .iter()
            .zip(&self.z_imag)
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .collect();
        softmax_inplace(&mut weights);
        weights
    }
}

// ── Softmax digital con sesgo CTP ────────────────────────────────────────────

/// Calcula A = Softmax(QK^T/√d + α·B) y Out = A V.
pub fn digital_softmax_attention(
    queries: &[Vec<f32>],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    ctp_bias: Option<&[Vec<f32>]>,
    alpha: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n = queries.len();
    let d_k = queries.first().map_or(1, |q| q.len()) as f32;
    let scale = d_k.sqrt().recip();
    let mut attention = vec![vec![0.0; n]; n];

    for i in 0..n {
        let mut logits = vec![0.0; n];
        for j in 0..n {
            let mut dot = 0.0f32;
            for d in 0..queries[i].len() {
                dot += queries[i][d] * keys[j][d];
            }
            logits[j] = dot * scale;
            if let Some(bias) = ctp_bias {
                logits[j] += alpha * bias[i][j];
            }
        }
        softmax_inplace(&mut logits);
        attention[i] = logits;
    }

    let d_v = values.first().map_or(0, |v| v.len());
    let mut output = vec![vec![0.0; d_v]; n];
    for i in 0..n {
        for j in 0..n {
            let a = attention[i][j];
            for d in 0..d_v {
                output[i][d] += a * values[j][d];
            }
        }
    }
    (output, attention)
}

// ── Motor híbrido principal ──────────────────────────────────────────────────

pub struct HybridThermoAttention {
    config: HybridThermoAttentionConfig,
    rff: PhasorRffMap,
    reservoir: LangevinReservoir,
    hybrid: NativeHybridPhasorCdtEngine,
    w_q: Vec<Vec<f32>>,
    w_k: Vec<Vec<f32>>,
    ctp_bias: Vec<Vec<f32>>,
    last_attention: Vec<Vec<f32>>,
    tick: u64,
}

impl HybridThermoAttention {
    pub fn new(config: HybridThermoAttentionConfig) -> Result<Self, HybridThermoAttentionError> {
        let d = config.d_model;
        let rff = PhasorRffMap::new(d, config.rff);
        let reservoir = LangevinReservoir::new(
            config.d_v,
            rff.features(),
            config.reservoir,
        );
        let hybrid = NativeHybridPhasorCdtEngine::new(
            NativeThermoCdtConfig {
                slices: 1,
                nodes_per_slice: config.cdt_nodes.max(16),
                spatial_degree: config.cdt_spatial_degree,
                temporal_degree: 1,
                temperature: 0.0,
                seed: config.cdt_seed,
                ..NativeThermoCdtConfig::default()
            },
            config.phasor,
            config.hybrid,
        )
        .map_err(|e| HybridThermoAttentionError::Hybrid(e.to_string()))?;

        Ok(Self {
            w_q: identity_matrix(d),
            w_k: identity_matrix(d),
            rff,
            reservoir,
            hybrid,
            ctp_bias: Vec::new(),
            last_attention: Vec::new(),
            tick: 0,
            config,
        })
    }

    pub fn config(&self) -> &HybridThermoAttentionConfig {
        &self.config
    }

    pub fn hybrid_engine(&self) -> &NativeHybridPhasorCdtEngine {
        &self.hybrid
    }

    pub fn hybrid_engine_mut(&mut self) -> &mut NativeHybridPhasorCdtEngine {
        &mut self.hybrid
    }

    pub fn ctp_bias(&self) -> &[Vec<f32>] {
        &self.ctp_bias
    }

    pub fn last_attention(&self) -> &[Vec<f32>] {
        &self.last_attention
    }

    /// Proyecta entradas crudas con W_Q, W_K aprendibles.
    pub fn project_qk(&self, raw: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let queries = raw
            .iter()
            .map(|x| matvec(&self.w_q, x))
            .collect::<Vec<_>>();
        let keys = raw.iter().map(|x| matvec(&self.w_k, x)).collect();
        (queries, keys)
    }

    /// Paso completo de inferencia híbrida sobre una secuencia.
    pub fn forward(
        &mut self,
        raw_tokens: &[Vec<f32>],
        values: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, HybridThermoAttentionReport), HybridThermoAttentionError> {
        if raw_tokens.is_empty() {
            return Err(HybridThermoAttentionError::EmptySequence);
        }
        let n = raw_tokens.len();
        self.validate_dims(raw_tokens, values)?;

        self.tick = self.tick.wrapping_add(1);
        let (queries, keys) = self.project_qk(raw_tokens);

        // Asegura que B_CTP tenga el tamaño correcto (ventana deslizante).
        self.resize_ctp_bias(n);

        // 1. Softmax digital con sesgo CTP.
        let (softmax_out, attention) = digital_softmax_attention(
            &queries,
            &keys,
            values,
            Some(&self.ctp_bias),
            self.config.ctp_coupling_alpha,
        );
        let softmax_entropy = mean_row_entropy(&attention);
        self.last_attention = attention.clone();

        // 2. Reservorio Langevin: acumula φ(K)†⊗V secuencialmente.
        self.reservoir.reset();
        let mut last_denom = 0.0f32;
        for j in 0..n {
            let (phi_r, phi_i) = self.rff.project(&keys[j]);
            for _ in 0..self.config.reservoir.steps_per_token.max(1) {
                self.reservoir.inject(&phi_r, &phi_i, &values[j]);
            }
        }

        // 3. Consulta del reservorio para cada Q (O(N) total).
        let mut reservoir_out = Vec::with_capacity(n);
        for i in 0..n {
            let (phi_r, phi_i) = self.rff.project(&queries[i]);
            let (out, denom) = self.reservoir.query(&phi_r, &phi_i);
            last_denom = denom;
            reservoir_out.push(out);
        }

        // 4. Inyecta A como frontera CTP y relaja en el motor híbrido.
        let cues = attention_to_boundary_cues(
            &attention,
            n,
            self.hybrid.core.node_count(),
            self.config.boundary_top_k,
            self.config.boundary_threshold,
            self.config.cdt_seed,
        );
        let wake = self
            .hybrid
            .infer_and_stage(&cues)
            .map_err(|e| HybridThermoAttentionError::Hybrid(e.to_string()))?;

        // 5. Extrae nuevo B_CTP del campo fasorial relajado.
        self.extract_ctp_bias(n);

        // 6. Mezcla Softmax + reservorio.
        let blend = self.config.thermo_blend.clamp(0.0, 1.0);
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = vec![0.0; self.config.d_v];
            for d in 0..self.config.d_v {
                row[d] = (1.0 - blend) * softmax_out[i][d] + blend * reservoir_out[i][d];
            }
            output.push(row);
        }

        // 7. Plasticidad Langevin-Hebbiana sobre W_Q, W_K.
        let plasticity_delta = self.plasticity_update(&attention, raw_tokens);

        let phasor_report = self.hybrid.phasor.report();
        Ok((
            output,
            HybridThermoAttentionReport {
                tick: self.tick,
                sequence_length: n,
                softmax_entropy,
                reservoir_denominator: last_denom,
                ctp_bias_norm: matrix_frobenius_norm(&self.ctp_bias),
                thermo_free_energy: phasor_report.free_energy,
                thermo_coherence: phasor_report.phase_coherence,
                plasticity_delta_norm: plasticity_delta,
                wake_gate_passed: wake.gate.passed,
                pending_attractors: wake.pending_count,
            },
        ))
    }

    /// Consolida atractores pendientes en el CDT (fase sleep).
    pub fn sleep_consolidate(
        &mut self,
    ) -> Result<
        crate::native_hybrid_phasor_cdt_engine::NativeHybridSleepReport,
        HybridThermoAttentionError,
    > {
        self.hybrid
            .sleep_consolidate()
            .map_err(|e| HybridThermoAttentionError::Hybrid(e.to_string()))
    }

    /// Extrae G_CTP(i,j) = Re{ψ_i* · ψ_j} del campo fasorial como sesgo futuro.
    fn extract_ctp_bias(&mut self, n: usize) {
        let nodes = self.hybrid.phasor.phasors.len();
        for i in 0..n {
            for j in 0..n {
                let ni = token_node(i, nodes, self.config.cdt_seed);
                let nj = token_node(j, nodes, self.config.cdt_seed);
                let pi = self.hybrid.phasor.phasors[ni];
                let pj = self.hybrid.phasor.phasors[nj];
                self.ctp_bias[i][j] = (pi.conj() * pj).re;
            }
        }
        // Normaliza para estabilidad numérica del Softmax.
        let norm = matrix_frobenius_norm(&self.ctp_bias).max(EPSILON);
        for row in &mut self.ctp_bias {
            for val in row {
                *val /= norm;
            }
        }
    }

    /// ΔW = -η ∇_W F_CTP + ruido térmico, aproximado por (A - P_eq) ⊗ x x^T.
    fn plasticity_update(&mut self, attention: &[Vec<f32>], raw: &[Vec<f32>]) -> f32 {
        let n = attention.len();
        if n == 0 {
            return 0.0;
        }
        let eq = self.reservoir.equilibrium_weights();
        let eta = self.config.plasticity_eta;
        let temp = self.config.plasticity_temperature;
        let seed = self.config.cdt_seed ^ self.tick;
        let mut delta_norm = 0.0f32;

        for i in 0..n {
            let error: f32 = attention[i]
                .iter()
                .zip(eq.iter().take(n))
                .map(|(&a, &e)| a - e)
                .sum::<f32>()
                / n as f32;

            for d in 0..self.config.d_model {
                let xi = raw[i][d];
                let noise = gaussian_from_counter(seed, (i as u64) * self.config.d_model as u64 + d as u64)
                    * temp.sqrt();
                let delta = -eta * error * xi + noise;
                for row in &mut self.w_q {
                    row[d] -= delta * 0.5;
                }
                for row in &mut self.w_k {
                    row[d] -= delta * 0.5;
                }
                delta_norm += delta * delta;
            }
        }
        delta_norm.sqrt()
    }

    fn validate_dims(
        &self,
        tokens: &[Vec<f32>],
        values: &[Vec<f32>],
    ) -> Result<(), HybridThermoAttentionError> {
        for t in tokens {
            if t.len() != self.config.d_model {
                return Err(HybridThermoAttentionError::DimensionMismatch {
                    expected: self.config.d_model,
                    got: t.len(),
                });
            }
        }
        for v in values {
            if v.len() != self.config.d_v {
                return Err(HybridThermoAttentionError::DimensionMismatch {
                    expected: self.config.d_v,
                    got: v.len(),
                });
            }
        }
        Ok(())
    }

    fn resize_ctp_bias(&mut self, n: usize) {
        if self.ctp_bias.len() == n && self.ctp_bias.first().map_or(0, |r| r.len()) == n {
            return;
        }
        self.ctp_bias = vec![vec![0.0; n]; n];
    }

    pub fn export_learned_state(&self) -> ThermoAttentionLearnedState {
        ThermoAttentionLearnedState {
            d_model: self.config.d_model,
            d_v: self.config.d_v,
            cdt_nodes: self.config.cdt_nodes,
            rff_features: self.config.rff.features,
            tick: self.tick,
            w_q: self.w_q.clone(),
            w_k: self.w_k.clone(),
            ctp_bias: self.ctp_bias.clone(),
            reservoir: ReservoirLearnedState {
                s_real: self.reservoir.s_real.clone(),
                s_imag: self.reservoir.s_imag.clone(),
                z_real: self.reservoir.z_real.clone(),
                z_imag: self.reservoir.z_imag.clone(),
                tick: self.reservoir.tick,
            },
            hybrid: self.hybrid.export_learned_state(),
        }
    }

    pub fn apply_learned_state(
        &mut self,
        state: &ThermoAttentionLearnedState,
    ) -> Result<(), HybridThermoAttentionError> {
        if state.d_model != self.config.d_model
            || state.d_v != self.config.d_v
            || state.cdt_nodes != self.config.cdt_nodes
            || state.rff_features != self.config.rff.features
        {
            return Err(HybridThermoAttentionError::DimensionMismatch {
                expected: self.config.d_model,
                got: state.d_model,
            });
        }
        self.tick = state.tick;
        self.w_q = state.w_q.clone();
        self.w_k = state.w_k.clone();
        self.ctp_bias = state.ctp_bias.clone();
        self.reservoir.s_real = state.reservoir.s_real.clone();
        self.reservoir.s_imag = state.reservoir.s_imag.clone();
        self.reservoir.z_real = state.reservoir.z_real.clone();
        self.reservoir.z_imag = state.reservoir.z_imag.clone();
        self.reservoir.tick = state.reservoir.tick;
        self.hybrid
            .apply_learned_state(&state.hybrid)
            .map_err(|error| HybridThermoAttentionError::Hybrid(error.to_string()))
    }

    /// Ajuste supervisado: acerca la salida termodinámica al hidden post-Transformer de Gemma.
    pub fn supervised_align_step(
        &mut self,
        raw_tokens: &[Vec<f32>],
        teacher_last: &[f32],
        learning_rate: f32,
    ) -> Result<f32, HybridThermoAttentionError> {
        if raw_tokens.is_empty() {
            return Err(HybridThermoAttentionError::EmptySequence);
        }
        let (output, _) = self.forward(raw_tokens, raw_tokens)?;
        let predicted = output
            .last()
            .ok_or(HybridThermoAttentionError::EmptySequence)?;
        let rate = learning_rate.clamp(1.0e-5, 0.25);
        let dims = predicted
            .len()
            .min(teacher_last.len())
            .min(self.config.d_model);
        let mut mse = 0.0f32;
        for dim in 0..dims {
            let error = teacher_last[dim] - predicted[dim];
            mse += error * error;
            let delta = rate * error * 0.01;
            for row in &mut self.w_q {
                if row.len() > dim {
                    row[dim] += delta * 0.5;
                }
            }
            for row in &mut self.w_k {
                if row.len() > dim {
                    row[dim] += delta * 0.5;
                }
            }
        }
        Ok(mse / dims.max(1) as f32)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermoAttentionLearnedState {
    pub d_model: usize,
    pub d_v: usize,
    pub cdt_nodes: usize,
    pub rff_features: usize,
    pub tick: u64,
    pub w_q: Vec<Vec<f32>>,
    pub w_k: Vec<Vec<f32>>,
    pub ctp_bias: Vec<Vec<f32>>,
    pub reservoir: ReservoirLearnedState,
    pub hybrid: HybridEngineLearnedState,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReservoirLearnedState {
    pub s_real: Vec<Vec<f32>>,
    pub s_imag: Vec<Vec<f32>>,
    pub z_real: Vec<f32>,
    pub z_imag: Vec<f32>,
    pub tick: u64,
}

// ── Utilidades ───────────────────────────────────────────────────────────────

fn dot_complex(a_r: &[f32], a_i: &[f32], b_r: &[f32], b_i: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for m in 0..a_r.len() {
        sum += a_r[m] * b_r[m] + a_i[m] * b_i[m];
    }
    sum
}

fn softmax_inplace(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.max(EPSILON).recip();
    for v in values.iter_mut() {
        *v *= inv;
    }
}

fn mean_row_entropy(attention: &[Vec<f32>]) -> f32 {
    if attention.is_empty() {
        return 0.0;
    }
    let n = attention.len();
    let mut total = 0.0f32;
    for row in attention {
        for &p in row {
            if p > EPSILON {
                total -= p * p.ln();
            }
        }
    }
    total / n as f32
}

fn matrix_frobenius_norm(matrix: &[Vec<f32>]) -> f32 {
    matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
}

fn identity_matrix(n: usize) -> Vec<Vec<f32>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

fn matvec(matrix: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector.iter())
                .map(|(&w, &x)| w * x)
                .sum()
        })
        .collect()
}

fn token_node(token_index: usize, node_count: usize, seed: u64) -> usize {
    splitmix64(seed ^ token_index as u64) as usize % node_count.max(1)
}

/// Convierte la matriz de atención en cues de frontera CTP para el motor híbrido.
fn attention_to_boundary_cues(
    attention: &[Vec<f32>],
    n: usize,
    node_count: usize,
    top_k: usize,
    threshold: f32,
    seed: u64,
) -> Vec<NativePhasorCue> {
    let mut cues = Vec::new();
    let k = top_k.min(n);
    for i in 0..n {
        let mut pairs: Vec<(usize, f32)> = attention[i]
            .iter()
            .enumerate()
            .filter(|(_, &a)| a >= threshold)
            .map(|(j, &a)| (j, a))
            .collect();
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        pairs.truncate(k);
        for (j, weight) in pairs {
            let node = token_node(j, node_count, seed ^ (i as u64).rotate_left(16));
            let phase = (weight * std::f32::consts::TAU).fract() * std::f32::consts::TAU;
            cues.push(NativePhasorCue {
                node,
                amplitude: weight.sqrt(),
                phase,
            });
        }
    }
    cues
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_sequence(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| signed_unit(seed ^ ((i as u64) << 16) ^ j as u64))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn rff_kernel_is_symmetric_and_bounded() {
        let rff = PhasorRffMap::new(8, PhasorRffConfig {
            features: 32,
            ..Default::default()
        });
        let x = synthetic_sequence(1, 8, 1)[0].clone();
        let y = synthetic_sequence(1, 8, 2)[0].clone();
        let k_xy = rff.kernel_approx(&x, &y);
        let k_yx = rff.kernel_approx(&y, &x);
        assert!((k_xy - k_yx).abs() < 1.0e-5);
        assert!(k_xy.abs() <= 1.0 + 1.0e-5);
    }

    #[test]
    fn softmax_attention_rows_sum_to_one() {
        let q = synthetic_sequence(4, 8, 10);
        let k = synthetic_sequence(4, 8, 11);
        let v = synthetic_sequence(4, 4, 12);
        let (_, a) = digital_softmax_attention(&q, &k, &v, None, 0.0);
        for row in &a {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn langevin_reservoir_query_is_finite() {
        let mut res = LangevinReservoir::new(4, 16, LangevinReservoirConfig::default());
        let rff = PhasorRffMap::new(8, PhasorRffConfig {
            features: 16,
            ..Default::default()
        });
        let k = synthetic_sequence(1, 8, 20)[0].clone();
        let v = vec![1.0, 0.0, -1.0, 0.5];
        let (phi_r, phi_i) = rff.project(&k);
        res.inject(&phi_r, &phi_i, &v);
        let q = synthetic_sequence(1, 8, 21)[0].clone();
        let (phi_r, phi_i) = rff.project(&q);
        let (out, denom) = res.query(&phi_r, &phi_i);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(denom.is_finite());
    }

    #[test]
    fn hybrid_forward_runs_and_updates_ctp_bias() {
        let config = HybridThermoAttentionConfig {
            d_model: 8,
            d_v: 4,
            rff: PhasorRffConfig {
                features: 16,
                ..Default::default()
            },
            cdt_nodes: 64,
            cdt_spatial_degree: 3,
            hybrid: NativeHybridConfig {
                minimizer: crate::native_phasor_thermodynamic_engine::NativePhasorMinimizerConfig {
                    max_iterations: 40,
                    residual_tolerance: 1.0e-2,
                    ..Default::default()
                },
                minimum_relative_energy_drop: 0.0,
                maximum_residual: 1.0e-1,
                minimum_magnetic_coherence: 0.5,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine = HybridThermoAttention::new(config).expect("engine");
        let tokens = synthetic_sequence(6, 8, 100);
        let values = synthetic_sequence(6, 4, 101);
        let (out, report) = engine.forward(&tokens, &values).expect("forward");
        assert_eq!(out.len(), 6);
        assert_eq!(out[0].len(), 4);
        assert!(report.ctp_bias_norm > 0.0);
        assert!(report.softmax_entropy > 0.0);
        assert_eq!(engine.ctp_bias().len(), 6);
    }

    #[test]
    fn ctp_bias_modulates_attention() {
        let q = synthetic_sequence(3, 4, 30);
        let k = synthetic_sequence(3, 4, 31);
        let v = synthetic_sequence(3, 2, 32);
        let (_, a_plain) = digital_softmax_attention(&q, &k, &v, None, 0.0);
        let mut bias = vec![vec![0.0; 3]; 3];
        bias[0][2] = 1.0;
        let (_, a_biased) = digital_softmax_attention(&q, &k, &v, Some(&bias), 2.0);
        assert!(a_biased[0][2] > a_plain[0][2]);
    }

    #[test]
    fn rff_similar_vectors_have_high_kernel() {
        let rff = PhasorRffMap::new(16, PhasorRffConfig {
            features: 128,
            sigma: 1.0,
            ..Default::default()
        });
        let x = synthetic_sequence(1, 16, 40)[0].clone();
        let mut y = x.clone();
        let self_kernel = rff.kernel_approx(&x, &y);
        y[0] += 0.5;
        let distant_kernel = rff.kernel_approx(&x, &y);
        assert!(self_kernel > distant_kernel);
        assert!(self_kernel > 0.5);
    }

    #[test]
    fn langevin_reservoir_memory_is_constant_in_sequence_length() {
        let features = 32;
        let d_v = 8;
        let rff = PhasorRffMap::new(8, PhasorRffConfig {
            features,
            ..Default::default()
        });
        let mut short = LangevinReservoir::new(d_v, features, LangevinReservoirConfig::default());
        let mut long = LangevinReservoir::new(d_v, features, LangevinReservoirConfig::default());
        let k = synthetic_sequence(1, 8, 50)[0].clone();
        let v = vec![1.0; d_v];
        let (phi_r, phi_i) = rff.project(&k);
        short.inject(&phi_r, &phi_i, &v);
        for _ in 0..64 {
            long.inject(&phi_r, &phi_i, &v);
        }
        assert_eq!(short.s_real.len(), features);
        assert_eq!(long.s_real.len(), features);
        assert_eq!(short.z_real.len(), features);
    }

    #[test]
    fn ctp_bias_persists_and_evolve_across_steps() {
        let config = test_hybrid_config();
        let mut engine = HybridThermoAttention::new(config).expect("engine");
        let tokens = synthetic_sequence(5, 8, 60);
        let values = synthetic_sequence(5, 4, 61);
        engine.forward(&tokens, &values).expect("step1");
        let bias1 = engine.ctp_bias()[0][1];
        engine.forward(&tokens, &values).expect("step2");
        let bias2 = engine.ctp_bias()[0][1];
        assert!(bias1.is_finite());
        assert!(bias2.is_finite());
        assert!(engine.ctp_bias().len() == 5);
    }

    #[test]
    fn hybrid_blend_interpolates_outputs() {
        let mut config = test_hybrid_config();
        config.thermo_blend = 0.0;
        let mut engine_soft = HybridThermoAttention::new(config).expect("soft");
        config.thermo_blend = 1.0;
        let mut engine_thermo = HybridThermoAttention::new(config).expect("thermo");
        let tokens = synthetic_sequence(4, 8, 70);
        let values = synthetic_sequence(4, 4, 71);
        let (out_soft, _) = engine_soft.forward(&tokens, &values).expect("soft");
        let (out_thermo, _) = engine_thermo.forward(&tokens, &values).expect("thermo");
        let diff: f32 = out_soft
            .iter()
            .flatten()
            .zip(out_thermo.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1.0e-4, "blend=0 y blend=1 deben diferir");
    }

    #[test]
    fn planted_needle_softmax_beats_uniform_baseline() {
        let (tokens, values, pairs) = planted_handshake_data(16, 8, 4, 300);
        let q = tokens.clone();
        let k = tokens.clone();
        let (_, attention) = digital_softmax_attention(&q, &k, &values, None, 0.0);
        let (top1, mrr) = handshake_metrics(&attention, &pairs);
        let uniform_top1 = 1.0 / tokens.len() as f32;
        assert!(top1 > uniform_top1, "top1={top1} uniform={uniform_top1}");
        assert!(mrr > uniform_top1);
    }

    #[test]
    fn last_attention_is_stored_after_forward() {
        let config = test_hybrid_config();
        let mut engine = HybridThermoAttention::new(config).expect("engine");
        let tokens = synthetic_sequence(5, 8, 80);
        let values = synthetic_sequence(5, 4, 81);
        engine.forward(&tokens, &values).expect("forward");
        assert_eq!(engine.last_attention().len(), 5);
        for row in engine.last_attention() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-4);
        }
    }

    #[test]
    fn plasticity_updates_change_projections() {
        let config = test_hybrid_config();
        let mut engine = HybridThermoAttention::new(config).expect("engine");
        let tokens = synthetic_sequence(8, 8, 90);
        let values = synthetic_sequence(8, 4, 91);
        let mut total_delta = 0.0f32;
        for _ in 0..5 {
            let (_, report) = engine.forward(&tokens, &values).expect("forward");
            total_delta += report.plasticity_delta_norm;
        }
        assert!(total_delta > 0.0, "plasticidad debe mover W_Q/W_K");
    }

    fn test_hybrid_config() -> HybridThermoAttentionConfig {
        HybridThermoAttentionConfig {
            d_model: 8,
            d_v: 4,
            rff: PhasorRffConfig {
                features: 16,
                ..Default::default()
            },
            cdt_nodes: 64,
            hybrid: NativeHybridConfig {
                minimizer: crate::native_phasor_thermodynamic_engine::NativePhasorMinimizerConfig {
                    max_iterations: 30,
                    residual_tolerance: 1.0e-2,
                    ..Default::default()
                },
                minimum_relative_energy_drop: 0.0,
                maximum_residual: 1.0e-1,
                minimum_magnetic_coherence: 0.5,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Secuencia con pares plantados: Q_i ≈ K_j para recuperación exacta.
    fn planted_handshake_data(
        n: usize,
        d: usize,
        pairs: usize,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<(usize, usize)>) {
        let mut tokens = synthetic_sequence(n, d, seed);
        let values = synthetic_sequence(n, 4.min(d), seed ^ 1);
        let mut planted = Vec::new();
        for p in 0..pairs.min(n / 2) {
            let q_idx = p * 2;
            let k_idx = p * 2 + 1;
            tokens[q_idx] = tokens[k_idx].clone();
            planted.push((q_idx, k_idx));
        }
        (tokens, values, planted)
    }

    fn handshake_metrics(attention: &[Vec<f32>], pairs: &[(usize, usize)]) -> (f32, f32) {
        if pairs.is_empty() {
            return (0.0, 0.0);
        }
        let mut top1 = 0usize;
        let mut mrr = 0.0f32;
        for &(q, k) in pairs {
            let row = &attention[q];
            let mass = row[k];
            let rank = 1 + row
                .iter()
                .enumerate()
                .filter(|(j, &a)| *j != k && a > mass)
                .count();
            if rank == 1 {
                top1 += 1;
            }
            mrr += 1.0 / rank as f32;
        }
        let n = pairs.len() as f32;
        (top1 as f32 / n, mrr / n)
    }

    #[test]
    fn sleep_consolidate_after_forward() {
        let config = HybridThermoAttentionConfig {
            d_model: 8,
            d_v: 4,
            rff: PhasorRffConfig {
                features: 16,
                ..Default::default()
            },
            cdt_nodes: 64,
            hybrid: NativeHybridConfig {
                minimizer: crate::native_phasor_thermodynamic_engine::NativePhasorMinimizerConfig {
                    max_iterations: 30,
                    residual_tolerance: 1.0e-2,
                    ..Default::default()
                },
                minimum_relative_energy_drop: 0.0,
                maximum_residual: 1.0e-1,
                minimum_magnetic_coherence: 0.5,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine = HybridThermoAttention::new(config).expect("engine");
        let tokens = synthetic_sequence(4, 8, 200);
        let values = synthetic_sequence(4, 4, 201);
        engine.forward(&tokens, &values).expect("forward");
        let sleep = engine.sleep_consolidate().expect("sleep");
        assert!(sleep.pending_before <= 1);
    }
}
