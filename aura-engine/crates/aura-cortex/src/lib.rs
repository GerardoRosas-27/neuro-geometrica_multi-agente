//! `aura-cortex`: Dos núcleos corticales complementarios de la propuesta:
//! 1. `DualCorticalCore` (AURA §3.2): Pesos Lentos Ternarios (planos de bits, 0-mul) +
//!    SSM Selectivo Continuo (por bit-shifts) + Pesos Rápidos (Regla Delta NLMS).
//! 2. `CorticalFabric` (Parte 1 Neuromórfica): Motor guiado por eventos SoA + CSR + búfer
//!    circular de retardos axónicos reales, atención por transporte paralelo en galga `U(1)`,
//!    Energía Libre de Laplace, competencia lateral en álgebra tropical (max-plus),
//!    aprendizaje local de 3 factores con histéresis ternaria y bucle neuromodulador (NE/ACh/DA).
//!
//! Cero `f32`/`f64` y cero multiplicaciones de matrices densas en el bucle caliente.

use aura_math::{
    build_cos_lut, decay_sym, decay_sym_i16, FastWeightMemory, LatentVec, TernaryMatrix,
    Xorshift64, LATENT_DIM, PHASE_STEPS,
};
use aura_perception::delta_modulate_byte;

pub const MAX_DELAY: usize = 16;
const MINICOLUMN_SIZE: usize = 8;

/// Estado neuromodulador global emitido por el núcleo metacognitivo (Parte 1 y AURA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Neuromodulators {
    pub noradrenaline: u8, // 0..=15: volatilidad / exploración / plasticidad rápida
    pub acetylcholine: i8, // -4..=+4: sesgo de precisión (basal sensorial vs apical prior)
    pub dopamine: i8,      // -1, 0, +1: tasa de reducción de energía libre (-dF)
}

impl Default for Neuromodulators {
    fn default() -> Self {
        Self {
            noradrenaline: 2,
            acetylcholine: 0,
            dopamine: 0,
        }
    }
}

/// Núcleo Cortical Dual de la arquitectura AURA (§3.2):
/// combina proyección ternaria sin multiplicaciones (`W_slow`), estado continuo SSM
/// con constante de tiempo selectiva por bit-shift (`h_t`) y memoria asociativa rápida (`M_t`).
pub struct DualCorticalCore {
    pub slow_proj: TernaryMatrix,
    pub gate_proj: TernaryMatrix,
    pub ssm_state: [i32; LATENT_DIM],
    pub fast_memory: FastWeightMemory,

    // Buffers reutilizables (cero asignaciones dinámicas en el hot loop).
    scratch_slow: [i32; LATENT_DIM],
    scratch_gate: [i32; LATENT_DIM],
    scratch_fast: [i32; LATENT_DIM],
    scratch_combined: [i32; LATENT_DIM],
}

impl DualCorticalCore {
    pub fn new(seed: u64, decay_shift: u8) -> Self {
        let mut rng = Xorshift64::from_seed(seed).split_stream(0x434F_5254);
        Self {
            slow_proj: TernaryMatrix::new_random_sparse(LATENT_DIM, LATENT_DIM, 20, &mut rng),
            gate_proj: TernaryMatrix::new_random_sparse(LATENT_DIM, LATENT_DIM, 15, &mut rng),
            ssm_state: [0; LATENT_DIM],
            fast_memory: FastWeightMemory::new(LATENT_DIM, decay_shift),
            scratch_slow: [0; LATENT_DIM],
            scratch_gate: [0; LATENT_DIM],
            scratch_fast: [0; LATENT_DIM],
            scratch_combined: [0; LATENT_DIM],
        }
    }

    pub fn reset_state(&mut self) {
        self.ssm_state.fill(0);
        self.fast_memory.reset();
    }

    /// Ejecuta un paso cortical dual sin asignaciones dinámicas.
    /// `ach > 0` acelera seguimiento sensorial; `ach < 0` prioriza inercia del estado SSM.
    pub fn step_cortex(&mut self, z_t: &LatentVec, ach: i8, use_fast: bool) -> LatentVec {
        // 1. Proyecciones ternarias con CERO multiplicaciones.
        self.slow_proj
            .project_zero_mul(&z_t.data, &mut self.scratch_slow);
        self.gate_proj
            .project_zero_mul(&z_t.data, &mut self.scratch_gate);

        // 2. Estado continuo SSM selectivo (constante de tiempo 2^shift elegida por entrada + ACh).
        for i in 0..LATENT_DIM {
            let target = self.scratch_slow[i] >> 2;
            let raw_gate = (self.scratch_gate[i] >> 7).clamp(-2, 2);
            // Mayor ACh -> menor shift (integra más rápido la entrada sensorial).
            let shift = (2 + raw_gate - (ach as i32)).clamp(1, 4) as u8;
            self.ssm_state[i] += (target - self.ssm_state[i]) >> shift;
        }

        // 3. Memoria asociativa de Pesos Rápidos (opcional para ablación).
        if use_fast {
            self.fast_memory.query(&z_t.data, &mut self.scratch_fast);
            for i in 0..LATENT_DIM {
                self.scratch_combined[i] = (self.ssm_state[i] >> 1) + (self.scratch_fast[i] >> 1);
            }
        } else {
            for i in 0..LATENT_DIM {
                self.scratch_combined[i] = self.ssm_state[i] >> 1;
            }
        }

        LatentVec::from_i32_shifted(&self.scratch_combined, 1)
    }

    /// Actualización online de la memoria rápida hacia el siguiente latente observado.
    pub fn adapt_fast(&mut self, key: &LatentVec, next_target: &LatentVec, lr_shift: u8) {
        let mut target_i32 = [0i32; LATENT_DIM];
        for i in 0..LATENT_DIM {
            target_i32[i] = (next_target.data[i] as i32) << 2;
        }
        self.fast_memory
            .update_online(&key.data, &target_i32, lr_shift);
    }
}

// ============================================================================
// PARTE 1 DE LA PROPUESTA: MOTOR NEUROMÓRFICO EVENT-DRIVEN (`CorticalFabric`)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynapseKind {
    BasalFeedforward, // Evidencia/error ascendente hacia dendrita basal
    ApicalFeedback,   // Predicción/contexto descendente hacia dendrita apical
}

/// Sinapsis compacta alineada: peso ternario `{-1, 0, +1}`, acumulador con histéresis,
/// conexión de galga `A_ij` en `Z_256`, escala log2 y retardo axónico real.
#[derive(Clone, Copy, Debug)]
pub struct Synapse {
    pub target: u32,
    pub weight: i8,      // -1, 0, +1 (cero multiplicación)
    pub latent_acc: i16, // Acumulador de plasticidad con histéresis
    pub phase_gauge: u8, // Conexión de galga A_ij en Z_256
    pub gain_shift: u8,  // Escala log2 (0..=4) -> desplazamiento de bits
    pub delay: u8,       // Retardo axonal real en ticks (1..MAX_DELAY)
    pub kind: SynapseKind,
}

/// Evento sináptico en vuelo en el búfer circular de retardos axónicos.
#[derive(Clone, Copy, Debug)]
struct DelayedArrival {
    syn_idx: u32,
    src_idx: u32,
    src_phase: u8,
}

/// Opciones de configuración / ablación controlada para `CorticalFabric`.
#[derive(Clone, Copy, Debug)]
pub struct FabricConfig {
    pub enable_plasticity: bool,
    pub enable_phase_gate: bool,
    pub enable_neuromod: bool,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            enable_plasticity: true,
            enable_phase_gate: true,
            enable_neuromod: true,
        }
    }
}

/// Telemetría devuelta en cada tick de `CorticalFabric::step`.
#[derive(Clone, Copy, Debug)]
pub struct FabricStepStats {
    pub free_energy: i64,
    pub sensory_free_energy: i64,
    pub spike_count: usize,
    pub neuromod: Neuromodulators,
}

/// Lámina cortical neuromórfica SoA + CSR guiada por eventos (0 `f32`/`f64`, 0 MatMul).
pub struct CorticalFabric {
    pub n_neurons: usize,
    pub v_basal: Vec<i32>,
    pub v_apical: Vec<i32>,
    pub error: Vec<i32>,
    pub threshold: Vec<i32>,
    pub phase: Vec<u8>,
    pub freq: Vec<u8>,
    pub precision_shift: Vec<u8>,
    pub trace: Vec<i16>,

    // Grafo disperso CSR
    pub row_ptr: Vec<usize>,
    pub synapses: Vec<Synapse>,

    // Búferes circulares (doble búfer: disparos y llegadas axónicas con retardo real)
    spike_ring: [Vec<u32>; MAX_DELAY],
    arrival_ring: [Vec<DelayedArrival>; MAX_DELAY],
    recent_arrivals: Vec<DelayedArrival>,
    drive_scratch: Vec<i32>,
    cos_lut: [i16; PHASE_STEPS],
    pub tick: usize,

    // Estado metacognitivo (NE, ACh, DA)
    pub prev_free_energy: i64,
    pub ema_free_energy: i64,
    pub fast_sensory_fe: i64,
    pub slow_sensory_fe: i64,
    pub neuromod: Neuromodulators,
    pub config: FabricConfig,
    rng: Xorshift64,
}

impl CorticalFabric {
    pub fn new(
        n_neurons: usize,
        row_ptr: Vec<usize>,
        synapses: Vec<Synapse>,
        freqs: Vec<u8>,
        seed: u64,
        config: FabricConfig,
    ) -> Self {
        assert_eq!(row_ptr.len(), n_neurons + 1);
        assert_eq!(freqs.len(), n_neurons);
        Self {
            n_neurons,
            v_basal: vec![0; n_neurons],
            v_apical: vec![0; n_neurons],
            error: vec![0; n_neurons],
            threshold: vec![192; n_neurons],
            phase: (0..n_neurons).map(|i| ((i * 13) & 0xFF) as u8).collect(),
            freq: freqs,
            precision_shift: vec![1; n_neurons],
            trace: vec![0; n_neurons],
            row_ptr,
            synapses,
            spike_ring: std::array::from_fn(|_| Vec::with_capacity(n_neurons / 8 + 8)),
            arrival_ring: std::array::from_fn(|_| Vec::with_capacity(n_neurons * 2)),
            recent_arrivals: Vec::with_capacity(n_neurons * 2),
            drive_scratch: vec![0; n_neurons],
            cos_lut: build_cos_lut(),
            tick: 0,
            prev_free_energy: 0,
            ema_free_energy: 0,
            fast_sensory_fe: 0,
            slow_sensory_fe: 0,
            neuromod: Neuromodulators::default(),
            config,
            rng: Xorshift64::from_seed(seed).split_stream(0x4641_4252),
        }
    }

    /// Construye una lámina jerárquica de 256 neuronas en 3 bandas de frecuencia:
    /// - Capa 0 (Sensorial Gamma, `0..64`, `freq = 17`): recibe bytes delta-modulados.
    /// - Capa 1 (Asociativa Beta, `64..192`, `freq = 9`): integra evidencia y proyecta predicción apical a Capa 0.
    /// - Capa 2 (Ápice Theta, `192..256`, `freq = 4`): contexto lento y predicción descendente a Capas 1 y 0.
    pub fn build_hierarchical_sheet(seed: u64, config: FabricConfig) -> Self {
        const N: usize = 256;
        let mut rng = Xorshift64::from_seed(seed).split_stream(0x4849_4552);
        let mut freqs = vec![0u8; N];
        for (i, f) in freqs.iter_mut().enumerate() {
            *f = if i < 64 {
                17 // Gamma
            } else if i < 192 {
                9 // Beta
            } else {
                4 // Theta
            };
        }

        let mut row_ptr = Vec::with_capacity(N + 1);
        let mut synapses = Vec::new();
        row_ptr.push(0);

        for src in 0..N {
            if src < 64 {
                // Feedforward sensorial -> asociativa (Capa 0 -> Capa 1) + eco local en Capa 0
                for _ in 0..8 {
                    let dst = 64 + rng.next_usize(128);
                    let w = if (rng.next_u64() & 1) == 0 { 1i8 } else { -1i8 };
                    let gauge = (rng.next_u64() & 0xFF) as u8;
                    let delay = 1 + (rng.next_u64() as u8 % 3);
                    synapses.push(Synapse {
                        target: dst as u32,
                        weight: w,
                        latent_acc: (w as i16) * 80,
                        phase_gauge: gauge,
                        gain_shift: 1,
                        delay,
                        kind: SynapseKind::BasalFeedforward,
                    });
                }
                for _ in 0..4 {
                    let dst = 16 + rng.next_usize(48);
                    if dst != src {
                        synapses.push(Synapse {
                            target: dst as u32,
                            weight: 1,
                            latent_acc: 80,
                            phase_gauge: (rng.next_u64() & 0xFF) as u8,
                            gain_shift: 2,
                            delay: 1 + (rng.next_u64() as u8 % 2),
                            kind: SynapseKind::BasalFeedforward,
                        });
                    }
                }
            } else if src < 192 {
                // Capa 1 (Beta): proyecta predicción apical descendente hacia Capa 0
                // (incluyendo las 16 neuronas sensoriales primarias) + feedforward a Capa 2.
                for _ in 0..8 {
                    let dst = rng.next_usize(64);
                    let init_w = match rng.next_u64() % 3 {
                        0 => 1i8,
                        1 => -1i8,
                        _ => 0i8,
                    };
                    let delay = 1 + (rng.next_u64() as u8 % 4);
                    synapses.push(Synapse {
                        target: dst as u32,
                        weight: init_w,
                        latent_acc: (init_w as i16) * 48,
                        phase_gauge: (rng.next_u64() & 0xFF) as u8,
                        gain_shift: 1,
                        delay,
                        kind: SynapseKind::ApicalFeedback,
                    });
                }
                for _ in 0..6 {
                    let dst = 192 + rng.next_usize(64);
                    let w = if (rng.next_u64() & 1) == 0 { 1i8 } else { -1i8 };
                    synapses.push(Synapse {
                        target: dst as u32,
                        weight: w,
                        latent_acc: (w as i16) * 80,
                        phase_gauge: (rng.next_u64() & 0xFF) as u8,
                        gain_shift: 1,
                        delay: 1 + (rng.next_u64() as u8 % 4),
                        kind: SynapseKind::BasalFeedforward,
                    });
                }
            } else {
                // Capa 2 (Theta): predicción apical descendente hacia Capa 1 y Capa 0
                for _ in 0..10 {
                    let dst = rng.next_usize(192);
                    let init_w = match rng.next_u64() % 3 {
                        0 => 1i8,
                        1 => -1i8,
                        _ => 0i8,
                    };
                    let delay = 2 + (rng.next_u64() as u8 % 6);
                    synapses.push(Synapse {
                        target: dst as u32,
                        weight: init_w,
                        latent_acc: (init_w as i16) * 48,
                        phase_gauge: (rng.next_u64() & 0xFF) as u8,
                        gain_shift: 1,
                        delay,
                        kind: SynapseKind::ApicalFeedback,
                    });
                }
            }
            row_ptr.push(synapses.len());
        }

        Self::new(N, row_ptr, synapses, freqs, seed, config)
    }

    /// Inyección SIN TOKENS: codifica un byte crudo por delta-modulación en neuronas `0..16`.
    pub fn inject_raw_byte(&mut self, prev_byte: u8, curr_byte: u8) {
        let (n, ids) = delta_modulate_byte(prev_byte, curr_byte);
        let slot = self.tick % MAX_DELAY;
        for &sensory_neuron in &ids[..n] {
            let idx = sensory_neuron as usize;
            if idx < self.n_neurons {
                self.v_basal[idx] = self.v_basal[idx].saturating_add(256).min(1024);
                self.trace[idx] = (self.trace[idx] + 64).min(256);
                self.spike_ring[slot].push(sensory_neuron);
            }
        }
    }

    /// Paso físico completo (0 floats, 0 MatMul):
    /// Propagación con retardos axónicos + Galga U(1) -> Energía Libre de Laplace +
    /// Competencia Tropical Max-Plus -> Plasticidad Local 3 Factores -> Metacognición NE/ACh/DA.
    pub fn step(&mut self) -> FabricStepStats {
        let curr_slot = self.tick % MAX_DELAY;
        let active_spikes = std::mem::take(&mut self.spike_ring[curr_slot]);
        let arriving = std::mem::take(&mut self.arrival_ring[curr_slot]);
        let spike_count = active_spikes.len();

        // 1A. Despachar nuevos disparos al búfer circular según el retardo axónico real `syn.delay`.
        for &src in &active_spikes {
            let src_idx = src as usize;
            let src_phase = self.phase[src_idx];
            let start = self.row_ptr[src_idx];
            let end = self.row_ptr[src_idx + 1];
            for syn_idx in start..end {
                let d = (self.synapses[syn_idx].delay as usize).clamp(1, MAX_DELAY - 1);
                let arr_slot = (self.tick + d) % MAX_DELAY;
                self.arrival_ring[arr_slot].push(DelayedArrival {
                    syn_idx: syn_idx as u32,
                    src_idx: src,
                    src_phase,
                });
            }
        }

        // 1B. Procesar eventos que llegan en este tick con atención por transporte paralelo U(1).
        self.recent_arrivals.clear();
        for ev in &arriving {
            let syn = self.synapses[ev.syn_idx as usize];
            self.recent_arrivals.push(*ev);
            if syn.weight == 0 {
                continue;
            }
            let dst = syn.target as usize;
            let mag = if self.config.enable_phase_gate {
                // Transporte paralelo de fase U(1): cos(theta_i + A_ij - theta_j)
                let phase_diff = ev
                    .src_phase
                    .wrapping_add(syn.phase_gauge)
                    .wrapping_sub(self.phase[dst]);
                let coherence = self.cos_lut[phase_diff as usize] as i32;
                if coherence <= 0 {
                    continue; // Canal cerrado automáticamente sin costo adicional
                }
                coherence >> syn.gain_shift
            } else {
                // Ablación sin compuerta de fase: ganancia media constante (128 >> gain_shift).
                128 >> syn.gain_shift
            };

            let impulse = if syn.weight > 0 { mag } else { -mag };
            match syn.kind {
                SynapseKind::BasalFeedforward => {
                    self.v_basal[dst] = (self.v_basal[dst] + impulse).clamp(-1024, 1024);
                }
                SynapseKind::ApicalFeedback => {
                    self.v_apical[dst] = (self.v_apical[dst] + impulse).clamp(-1024, 1024);
                }
            }
        }

        // 2. Dinámica dendrítica, Energía Libre de Laplace y cálculo de drive por neurona.
        let mut total_free_energy: i64 = 0;
        let mut sensory_free_energy: i64 = 0;
        let ach = if self.config.enable_neuromod {
            self.neuromod.acetylcholine
        } else {
            0
        };
        let ne_level = if self.config.enable_neuromod {
            self.neuromod.noradrenaline
        } else {
            1
        };

        for i in 0..self.n_neurons {
            // Fuga pasiva exponencial simétrica por bit-shift.
            self.v_basal[i] = decay_sym(self.v_basal[i], 3);
            self.v_apical[i] = decay_sym(self.v_apical[i], 3);
            self.trace[i] = decay_sym_i16(self.trace[i], 3);

            // Error de predicción objetivo (misma definición en todos los brazos para comparación justa).
            let raw_err = self.v_basal[i] - self.v_apical[i];
            self.error[i] = raw_err;
            let weighted_fe = raw_err.abs() >> self.precision_shift[i];
            total_free_energy += weighted_fe as i64;
            if i < 64 {
                sensory_free_energy += weighted_fe as i64;
            }

            // Oscilador de fase + ruido exploratorio por Noradrenalina (solo cuando NE es alta).
            let jitter = if ne_level >= 5 {
                let shift = 8u8.saturating_sub(ne_level >> 1).max(3);
                (self.rng.next_signed_byte() >> shift) as u8
            } else {
                0u8
            };
            self.phase[i] = self.phase[i].wrapping_add(self.freq[i]).wrapping_add(jitter);

            // Acetilcolina modula el balance de drive basal vs apical (sin distorsionar la FE medida).
            let basal_drive = if ach > 0 {
                self.v_basal[i] + (self.v_basal[i] >> 3)
            } else {
                self.v_basal[i]
            };
            let apical_drive = if ach < 0 {
                self.v_apical[i] + (self.v_apical[i] >> 3)
            } else {
                self.v_apical[i]
            };
            let coincidence = (basal_drive + apical_drive) >> 1;
            self.drive_scratch[i] = coincidence.max(weighted_fe);
        }

        // 2B. Competencia lateral en Álgebra Tropical (Max-Plus) por minicolumna cortical
        // y disparo homeostático.
        let next_slot = (self.tick + 1) % MAX_DELAY;
        let n_cols = self.n_neurons / MINICOLUMN_SIZE;
        for col in 0..n_cols {
            let base = col * MINICOLUMN_SIZE;
            let mut max_drive = i32::MIN;
            let mut winner = base;
            for k in base..(base + MINICOLUMN_SIZE) {
                if self.drive_scratch[k] > max_drive {
                    max_drive = self.drive_scratch[k];
                    winner = k;
                }
            }
            let lateral_floor = max_drive - 48; // Semianillo tropical (max, +) con phi_ij = -48
            for k in base..(base + MINICOLUMN_SIZE) {
                let d = self.drive_scratch[k];
                let wins_competition = k == winner || d >= lateral_floor;
                if wins_competition && d > self.threshold[k] {
                    self.spike_ring[next_slot].push(k as u32);
                    self.v_basal[k] >>= 1;
                    self.trace[k] = (self.trace[k] + 64).min(256);
                    self.threshold[k] = (self.threshold[k] + 12).min(1024);
                } else {
                    self.threshold[k] = (self.threshold[k] - 2).max(96);
                }
            }
        }

        // 3. Plasticidad local de 3 factores (e-prop sin backprop) en sinapsis activas.
        if self.config.enable_plasticity {
            let da = if self.config.enable_neuromod {
                self.neuromod.dopamine
            } else {
                0
            };
            let step_mag: i16 = if ne_level >= 5 {
                12
            } else if da > 0 {
                7
            } else {
                5
            };
            let gauge_step: u8 = if ne_level >= 5 { 4 } else { 1 };

            for ev in &self.recent_arrivals {
                let pre_trace = self.trace[ev.src_idx as usize];
                if pre_trace <= 0 {
                    continue;
                }
                let syn = &mut self.synapses[ev.syn_idx as usize];
                let dst = syn.target as usize;
                let post_err = self.error[dst];
                if post_err.abs() < 8 {
                    continue;
                }

                match syn.kind {
                    SynapseKind::ApicalFeedback => {
                        let err_sign = post_err.signum() as i16;
                        let delta = err_sign * step_mag;
                        syn.latent_acc = syn.latent_acc.saturating_add(delta).clamp(-256, 256);

                        if syn.latent_acc > 64 {
                            syn.weight = 1;
                        } else if syn.latent_acc < -64 {
                            syn.weight = -1;
                        } else if syn.latent_acc.abs() < 16 {
                            syn.weight = 0;
                        }

                        if self.config.enable_phase_gate && syn.weight != 0 {
                            let target_gauge = self.phase[dst].wrapping_sub(ev.src_phase);
                            let diff = target_gauge.wrapping_sub(syn.phase_gauge) as i8;
                            let helpful = (syn.weight as i32) * post_err.signum() > 0;
                            if helpful {
                                if diff > 0 {
                                    syn.phase_gauge = syn.phase_gauge.wrapping_add(gauge_step);
                                } else if diff < 0 {
                                    syn.phase_gauge = syn.phase_gauge.wrapping_sub(gauge_step);
                                }
                            } else {
                                syn.phase_gauge = syn.phase_gauge.wrapping_add(gauge_step);
                            }
                        }
                    }
                    SynapseKind::BasalFeedforward => {
                        if da != 0 && self.config.enable_phase_gate {
                            let target_gauge = self.phase[dst].wrapping_sub(ev.src_phase);
                            let diff = target_gauge.wrapping_sub(syn.phase_gauge) as i8;
                            if diff > 0 {
                                syn.phase_gauge = syn.phase_gauge.wrapping_add(1);
                            } else if diff < 0 {
                                syn.phase_gauge = syn.phase_gauge.wrapping_sub(1);
                            }
                        }
                    }
                }
            }
        }

        // 4. Núcleo metacognitivo de segundo orden (NE, ACh, DA).
        self.update_metacognition(total_free_energy, sensory_free_energy, spike_count);

        self.spike_ring[curr_slot] = active_spikes;
        self.spike_ring[curr_slot].clear();
        self.arrival_ring[curr_slot] = arriving;
        self.arrival_ring[curr_slot].clear();
        self.tick += 1;

        FabricStepStats {
            free_energy: total_free_energy,
            sensory_free_energy,
            spike_count,
            neuromod: self.neuromod,
        }
    }

    fn update_metacognition(&mut self, current_fe: i64, sensory_fe: i64, spike_count: usize) {
        if self.tick == 0 {
            self.ema_free_energy = current_fe;
            self.fast_sensory_fe = sensory_fe;
            self.slow_sensory_fe = sensory_fe;
            self.prev_free_energy = current_fe;
            return;
        }
        self.ema_free_energy += (current_fe - self.ema_free_energy) >> 4;
        // Doble escala temporal (Yu & Dayan): EMA rápida (8 ticks) vs EMA lenta (64 ticks)
        // suaviza el reloj periódico de llegada de bytes y dispara NE solo ante cambios de régimen.
        self.fast_sensory_fe += (sensory_fe - self.fast_sensory_fe) >> 3;
        self.slow_sensory_fe += (sensory_fe - self.slow_sensory_fe) >> 6;

        let delta_fe = self.prev_free_energy - current_fe;
        self.neuromod.dopamine = if delta_fe > 32 {
            1
        } else if delta_fe < -32 {
            -1
        } else {
            0
        };

        let volatility = self.fast_sensory_fe - self.slow_sensory_fe;
        if volatility > (self.slow_sensory_fe >> 3).max(40) {
            self.neuromod.noradrenaline = (self.neuromod.noradrenaline + 3).min(15);
        } else {
            self.neuromod.noradrenaline = self.neuromod.noradrenaline.saturating_sub(1);
        }

        let target_activity = (self.n_neurons >> 4).max(4);
        if volatility > (self.slow_sensory_fe >> 4).max(24) && spike_count < target_activity {
            self.neuromod.acetylcholine = (self.neuromod.acetylcholine + 1).min(2);
        } else if volatility < 0 {
            self.neuromod.acetylcholine = (self.neuromod.acetylcholine - 1).max(-2);
        }

        self.prev_free_energy = current_fe;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_cortical_core_deterministic_and_bounded() {
        let mut c1 = DualCorticalCore::new(0xA400, 6);
        let mut c2 = DualCorticalCore::new(0xA400, 6);
        let mut z = LatentVec::zero();
        for i in 0..LATENT_DIM {
            z.data[i] = ((i as i32 * 17) % 120 - 60) as i8;
        }
        let o1 = c1.step_cortex(&z, 1, true);
        let o2 = c2.step_cortex(&z, 1, true);
        assert_eq!(o1, o2);
        assert!(o1.l1_norm() > 0);
    }

    #[test]
    fn cortical_fabric_reduces_free_energy_and_keeps_sparsity() {
        let mut plastic = CorticalFabric::build_hierarchical_sheet(0xA400, FabricConfig::default());
        let mut frozen = CorticalFabric::build_hierarchical_sheet(
            0xA400,
            FabricConfig {
                enable_plasticity: false,
                ..FabricConfig::default()
            },
        );

        let motif = [0x55u8, 0xAA, 0x33, 0xCC, 0x0F, 0xF0];
        let mut prev = 0u8;
        let mut fe_plastic_late = 0i64;
        let mut fe_frozen_late = 0i64;
        let mut spikes_total = 0usize;

        for t in 0..480 {
            let b = motif[t % motif.len()];
            plastic.inject_raw_byte(prev, b);
            frozen.inject_raw_byte(prev, b);
            let sp = plastic.step();
            let sf = frozen.step();
            prev = b;
            spikes_total += sp.spike_count;
            if t >= 320 {
                fe_plastic_late += sp.free_energy;
                fe_frozen_late += sf.free_energy;
            }
        }
        let mean_spikes = spikes_total as f64 / 480.0;
        assert!(
            mean_spikes >= 2.0 && mean_spikes <= 96.0,
            "cortical activity out of homeostatic band: {mean_spikes}"
        );
        assert!(
            fe_plastic_late < fe_frozen_late,
            "expected plastic fabric FE ({fe_plastic_late}) < frozen FE ({fe_frozen_late})"
        );
    }
}
