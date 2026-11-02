//! Módulo de Hipocampo Híbrido, Neuromodulación y Consolidación SWR (Sharp-Wave Ripples)
//!
//! Fusiona la Teoría de Sistemas de Aprendizaje Complementarios (CLS: McClelland et al., 1995;
//! Kumaran et al., 2016) con el búfer de trayectorias CDT (Curvature-guided Dynamical Trajectories)
//! de Constelación v4:
//! 1. Giro Dentado (DG): Separación de patrones ultra-dispersa (k-WTA).
//! 2. CA3: Memoria episódica continua (Hopfield / Engramas) de operadores de transición dinámica.
//! 3. CA1: Comparador entorrinal-hipocampal de novedad y sorpresa.
//! 4. Neuromoduladores biológicos: Dopamina (DA), Acetilcolina (ACh), Noradrenalina (NE), Serotonina (5-HT)
//!    coordinados por reloj oscilatorio Theta-Gamma (Buzsáki, 2006).
//! 5. Consolidación del Sueño SWR: Muestreo priorizado por sorpresa/error que destila los pesos
//!    volátiles rápidos ($W^F$, BTSP) a los pesos sinápticos lentos ($W^P$, Dale) y poda sinapsis inactivas.

use crate::rng::Xoshiro256StarStar;
use std::f64::consts::PI;

/// Estado de los cuatro neuromoduladores primarios del cerebro bioinspirado
/// y reloj de fase oscilatoria Theta (4-8 Hz) / Gamma (30-80 Hz).
#[derive(Debug, Clone)]
pub struct Neuromodulators {
    /// Dopamina (DA): Señal de error de predicción y saliencia (Schultz et al., 1997).
    /// Regula la tasa de plasticidad LTP/LTD en capas corticales y estriatales.
    pub da: f64,
    /// Acetilcolina (ACh): Señal de novedad y atención sensorial (Hasselmo, 2006).
    /// Alta en vigilia/exploración (suprime transmisión recurrente, maximiza plasticidad feedforward).
    /// Baja en sueño/consolidación (facilita reactivación recurrente CA3 y SWR).
    pub ach: f64,
    /// Noradrenalina (NE): Incertidumbre de estado e hiperalerta (Aston-Jones & Cohen, 2005).
    /// Modula la ganancia no lineal de las columnas somáticas y el umbral de disparo.
    pub ne: f64,
    /// Serotonina (5-HT): Descuento temporal, paciencia y homeostasis inhibitoria (Cools et al., 2008).
    pub serotonin: f64,
    /// Fase oscilatoria Theta en radianes [0, 2π).
    pub theta_phase: f64,
    /// Subciclo Gamma dentro del ciclo Theta actual (0..4).
    pub gamma_subcycle: usize,
}

impl Default for Neuromodulators {
    fn default() -> Self {
        Self {
            da: 0.1,
            ach: 0.8, // Por defecto en modo vigilia/atención
            ne: 0.2,
            serotonin: 0.5,
            theta_phase: 0.0,
            gamma_subcycle: 0,
        }
    }
}

impl Neuromodulators {
    /// Avanza el reloj oscilatorio Theta-Gamma un micro-paso de tiempo.
    pub fn step_clock(&mut self) {
        self.theta_phase += 2.0 * PI / 16.0;
        if self.theta_phase >= 2.0 * PI {
            self.theta_phase -= 2.0 * PI;
            self.gamma_subcycle = 0;
        } else {
            self.gamma_subcycle = (self.theta_phase / (PI / 2.0)) as usize;
            if self.gamma_subcycle > 3 {
                self.gamma_subcycle = 3;
            }
        }
    }

    /// Actualiza niveles de neuromoduladores en función del error relativo y novedad observada.
    pub fn update_signals(&mut self, rel_err: f64, surprise: f64, success: bool) {
        // DA responde al éxito en predicción (RPE)
        let rpe = if success {
            0.4
        } else {
            -0.2 * rel_err.min(3.0)
        };
        self.da = (self.da * 0.8 + 0.1 + rpe).clamp(0.01, 2.0);

        // ACh sube con la novedad/sorpresa ambiental
        let novelty_drive = (surprise / (1.0 + surprise)).clamp(0.0, 1.0);
        self.ach = (self.ach * 0.7 + 0.3 * (0.2 + 0.8 * novelty_drive)).clamp(0.05, 1.5);

        // NE sube con el error repentino o incertidumbre alta
        self.ne = (self.ne * 0.75 + 0.25 * rel_err.min(2.0)).clamp(0.05, 1.8);

        // Serotonina promueve estabilidad tónica
        self.serotonin = (self.serotonin * 0.95 + 0.05 * 0.5).clamp(0.1, 1.0);
    }

    /// Ajusta neuromoduladores al entrar en fase de sueño/consolidación SWR.
    pub fn enter_sleep_mode(&mut self) {
        self.ach = 0.05; // Supresión colinérgica: permite auto-reactivación CA3
        self.da = 0.05;
        self.ne = 0.02; // Mínimo tono adrenérgico en sueño de ondas lentas
        self.serotonin = 0.8; // Alta serotonina estabilizadora
    }

    /// Restaura neuromoduladores a estado de vigilia activa.
    pub fn enter_wake_mode(&mut self) {
        self.ach = 0.85;
        self.da = 0.20;
        self.ne = 0.25;
        self.serotonin = 0.50;
    }
}

/// Engrama episódico almacenado en el autoasociador CA3.
/// Almacena el vector de contexto de la tarea, la clave del Giro Dentado y
/// el operador de adaptación rápida consolidado en la ventana temporal.
#[derive(Debug, Clone)]
pub struct EpisodicEngram {
    /// Vector de contexto promediado (ej. 10 floats de embedding contextual)
    pub context: [f64; 10],
    /// Representación dispersa k-WTA en el Giro Dentado
    pub dg_sparse: Vec<f64>,
    /// Parámetros del operador de adaptación rápida W^F descubiertos en el episodio
    pub wf_operator: Vec<f64>,
    /// Saliencia del engrama (error de predicción inicial + recompensa)
    pub salience: f64,
    /// Edad en episodios
    pub age: usize,
    /// Número de veces reactivado
    pub access_count: usize,
}

/// Registro de transición en el búfer dinámico CDT (Curvature Dynamical Trajectories)
#[derive(Debug, Clone)]
pub struct CdtTransition {
    /// Vector de contexto del episodio
    pub context: [f64; 10],
    /// Coordenadas de los 6 puntos en t: [x0, y0, ..., x5, y5]
    pub q_t: Vec<f64>,
    /// Coordenadas de los 6 puntos en t+1: [x0', y0', ..., x5', y5']
    pub q_next: Vec<f64>,
    /// Familia dinámica a la que pertenece la trayectoria (0..11)
    pub family_id: usize,
    /// Error relativo de la predicción en el paso
    pub rel_err: f64,
    /// Sorpresa o novedad local calculada por CA1
    pub surprise: f64,
}

/// Hipocampo Híbrido bio-inspirado con búfer CDT y sueño SWR.
#[derive(Debug, Clone)]
pub struct HybridHippocampusCdt {
    /// Sistema de neuromodulación y reloj
    pub neuromod: Neuromodulators,
    /// Dimensión del Giro Dentado (DG)
    pub dg_dim: usize,
    /// Matriz de proyección sensorial a DG (dale-compliant excitatoria fija)
    pub dg_weights: Vec<f64>,
    /// Memoria episódica CA3 (red autoasociativa continua)
    pub ca3_engrams: Vec<EpisodicEngram>,
    /// Capacidad máxima de engramas en CA3
    pub max_engrams: usize,
    /// Búfer de transiciones CDT para repetición rápida y consolidación
    pub cdt_buffer: Vec<CdtTransition>,
    /// Capacidad máxima del búfer CDT
    pub max_cdt_buffer: usize,
    /// Tasa de consolidación sináptica en sueño (Frey-Morris tagging)
    pub consolidation_rate: f64,
}

impl HybridHippocampusCdt {
    /// Construye una nueva estructura hipocampal híbrida.
    pub fn new(
        dg_dim: usize,
        max_engrams: usize,
        max_cdt_buffer: usize,
        consolidation_rate: f64,
        rng: &mut Xoshiro256StarStar,
    ) -> Self {
        // Inicializar proyección a DG con distribución positiva semi-ortogonal
        let mut dg_weights = vec![0.0; 10 * dg_dim];
        for val in dg_weights.iter_mut() {
            *val = (rng.randn().abs() * 0.4).min(1.5);
        }

        Self {
            neuromod: Neuromodulators::default(),
            dg_dim,
            dg_weights,
            ca3_engrams: Vec::with_capacity(max_engrams),
            max_engrams,
            cdt_buffer: Vec::with_capacity(max_cdt_buffer),
            max_cdt_buffer,
            consolidation_rate,
        }
    }

    /// Giro Dentado (DG): Separación de patrones mediante proyección lineal dispersada por k-WTA.
    /// Selecciona exactamente los `k` canales más activos, anulando el resto a 0.0 (inhibición feedforward PV+).
    pub fn encode_dg(&self, context: &[f64; 10], k_active: usize) -> Vec<f64> {
        let mut raw_activations = vec![0.0; self.dg_dim];
        for d in 0..self.dg_dim {
            let mut sum = 0.0;
            for c in 0..10 {
                sum += context[c] * self.dg_weights[c * self.dg_dim + d];
            }
            raw_activations[d] = sum.max(0.0);
        }

        // k-WTA: Encontrar el umbral del top-k
        let mut sorted = raw_activations.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = if k_active < sorted.len() {
            sorted[k_active]
        } else {
            0.0
        };

        let mut sparse_code = vec![0.0; self.dg_dim];
        let mut active_count = 0;
        let mut norm_sq = 0.0;

        for d in 0..self.dg_dim {
            if raw_activations[d] >= threshold
                && active_count < k_active
                && raw_activations[d] > 1e-6
            {
                sparse_code[d] = raw_activations[d];
                norm_sq += sparse_code[d] * sparse_code[d];
                active_count += 1;
            }
        }

        // Normalizar L2 para estabilidad de consulta en CA3
        let norm = norm_sq.sqrt();
        if norm > 1e-7 {
            for v in sparse_code.iter_mut() {
                *v /= norm;
            }
        }

        sparse_code
    }

    /// CA3: Recuperación de engrama episódico por similitud coseno continua con softmax.
    /// Si la similitud supera `min_sim`, retorna una combinación interpolada del operador W^F y su similitud.
    pub fn recall_ca3(&mut self, dg_code: &[f64], min_sim: f64) -> Option<(Vec<f64>, f64)> {
        if self.ca3_engrams.is_empty() {
            return None;
        }

        let mut similarities = Vec::with_capacity(self.ca3_engrams.len());
        let mut max_sim = -1.0f64;
        let mut best_idx = 0;

        for (idx, engram) in self.ca3_engrams.iter().enumerate() {
            let mut dot = 0.0;
            for (a, b) in dg_code.iter().zip(engram.dg_sparse.iter()) {
                dot += *a * *b;
            }
            if dot > max_sim {
                max_sim = dot;
                best_idx = idx;
            }
            similarities.push(dot);
        }

        if max_sim < min_sim {
            return None;
        }

        // Registrar acceso
        self.ca3_engrams[best_idx].access_count += 1;

        // Recuperar operador con softmax térmico (temperatura T=0.2)
        let temp = 0.2;
        let mut exp_weights = Vec::with_capacity(self.ca3_engrams.len());
        let mut sum_exp = 0.0;

        for &sim in &similarities {
            if sim > min_sim {
                let w = ((sim - max_sim) / temp).exp();
                exp_weights.push(w);
                sum_exp += w;
            } else {
                exp_weights.push(0.0);
            }
        }

        let op_len = self.ca3_engrams[best_idx].wf_operator.len();
        let mut blended_op = vec![0.0; op_len];

        if sum_exp > 1e-8 {
            for (idx, engram) in self.ca3_engrams.iter().enumerate() {
                let w = exp_weights[idx] / sum_exp;
                if w > 1e-5 {
                    for (dst, src) in blended_op.iter_mut().zip(engram.wf_operator.iter()) {
                        *dst += w * *src;
                    }
                }
            }
        } else {
            blended_op.copy_from_slice(&self.ca3_engrams[best_idx].wf_operator);
        }

        Some((blended_op, max_sim))
    }

    /// Almacena o actualiza un engrama episódico en CA3.
    /// Si CA3 está lleno, reemplaza el engrama menos saliente y más antiguo.
    pub fn store_ca3(
        &mut self,
        context: [f64; 10],
        dg_sparse: Vec<f64>,
        wf_operator: Vec<f64>,
        salience: f64,
    ) {
        // Envejecer engramas existentes
        for e in self.ca3_engrams.iter_mut() {
            e.age += 1;
        }

        if self.ca3_engrams.len() < self.max_engrams {
            self.ca3_engrams.push(EpisodicEngram {
                context,
                dg_sparse,
                wf_operator,
                salience,
                age: 0,
                access_count: 1,
            });
        } else {
            // Reemplazo: encontrar engrama con menor puntuación de retención
            // retención = saliencia * sqrt(accesos) / (1 + 0.1 * edad)
            let mut min_retention = f64::MAX;
            let mut victim_idx = 0;

            for (idx, e) in self.ca3_engrams.iter().enumerate() {
                let ret =
                    e.salience * (e.access_count as f64).sqrt() / (1.0 + 0.05 * (e.age as f64));
                if ret < min_retention {
                    min_retention = ret;
                    victim_idx = idx;
                }
            }

            self.ca3_engrams[victim_idx] = EpisodicEngram {
                context,
                dg_sparse,
                wf_operator,
                salience,
                age: 0,
                access_count: 1,
            };
        }
    }

    /// Empuja una transición dinámica al búfer CDT.
    pub fn push_cdt(&mut self, trans: CdtTransition) {
        if self.cdt_buffer.len() >= self.max_cdt_buffer {
            // Si está lleno, eliminar el primer elemento (FIFO simple)
            self.cdt_buffer.remove(0);
        }
        self.cdt_buffer.push(trans);
    }

    /// Muestra un mini-lote de transiciones priorizadas por sorpresa/error relativo
    /// para el ciclo de reactivación durante Sharp-Wave Ripples (SWR).
    pub fn sample_swr_batch(
        &self,
        batch_size: usize,
        rng: &mut Xoshiro256StarStar,
    ) -> Vec<CdtTransition> {
        let n = self.cdt_buffer.len();
        if n == 0 {
            return Vec::new();
        }
        if n <= batch_size {
            return self.cdt_buffer.clone();
        }

        // Ponderación priorizada: p_i ~ (surprise_i + 0.5 * rel_err_i + 0.05)
        let mut cdf = Vec::with_capacity(n);
        let mut accum = 0.0;
        for t in &self.cdt_buffer {
            let priority = t.surprise.max(0.0) + 0.5 * t.rel_err.clamp(0.0, 5.0) + 0.05;
            accum += priority;
            cdf.push(accum);
        }

        let mut samples = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let target = rng.gen_f64() * accum;
            // Búsqueda binaria
            let idx = match cdf
                .binary_search_by(|v| v.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i,
                Err(i) => i.min(n - 1),
            };
            samples.push(self.cdt_buffer[idx].clone());
        }

        samples
    }
}
