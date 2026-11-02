//! Módulo de Fasores Complejos y Handshake de Wheeler-Feynman
//!
//! Implementa el principio físico descubierto en `feature/colapso-honda-Handshake`
//! y `feature/motor-termodinamico-fasores`:
//! Inferencia rápida guiada por interferencia de ondas adelantadas (futuras)
//! y retardadas (presentes) en el espacio de Hilbert fasorial $\mathbb{C}^d$.
//!
//! La acción de Handshake:
//! $$\mathcal{A}_{\text{handshake}} = -\sum_{j=1}^d \Re(\psi_{\text{ret}, j}^* \cdot \psi_{\text{adv}, j})$$
//! minimiza la energía libre $\mathcal{F} = \mathcal{U} - T\mathcal{S}$ colapsando la
//! trayectoria en un atractor coherente sin explosión combinatoria.


/// Número complejo simple en f64 para representar fasores de amplitud y fase
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexPhasor {
    pub re: f64,
    pub im: f64,
}

impl ComplexPhasor {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

/// Motor de Interferencia y Colapso Transaccional Wheeler-Feynman
#[derive(Debug, Clone)]
pub struct WheelerFeynmanHandshakeEngine {
    /// Dimensión del espacio fasorial
    pub dim: usize,
    /// Temperatura termodinámica para descenso de energía libre ($T$)
    pub temperature: f64,
    /// Frecuencias intrínsecas de acoplamiento fasorial $\omega_j$
    pub frequencies: Vec<f64>,
    /// Acoplamientos simétricos inter-fasor $J_{jk}$
    pub coupling_matrix: Vec<Vec<f64>>,
}

impl WheelerFeynmanHandshakeEngine {
    pub fn new(dim: usize, temperature: f64) -> Self {
        let mut frequencies = Vec::with_capacity(dim);
        for j in 0..dim {
            let f = 0.1 + 0.9 * ((j as f64 * 1.6180339887) % 1.0);
            frequencies.push(f);
        }

        // Matriz de acoplamiento simétrica con decaimiento de distancia
        let mut coupling_matrix = vec![vec![0.0; dim]; dim];
        for j in 0..dim {
            for k in 0..dim {
                if j == k {
                    coupling_matrix[j][k] = 1.0;
                } else {
                    let dist = (j as isize - k as isize).abs() as f64;
                    coupling_matrix[j][k] = 0.2 / (1.0 + dist * 0.5);
                }
            }
        }

        Self {
            dim,
            temperature,
            frequencies,
            coupling_matrix,
        }
    }

    /// Construye la onda retardada $\psi_{\text{ret}}(t)$ desde la observación presente
    pub fn build_retarded_wave(&self, input_features: &[f64], t: f64) -> Vec<ComplexPhasor> {
        let mut psi_ret = Vec::with_capacity(self.dim);
        for j in 0..self.dim {
            let val = if j < input_features.len() {
                input_features[j]
            } else {
                0.0
            };
            let amp = val.tanh();
            let phase = self.frequencies[j] * t;
            psi_ret.push(ComplexPhasor::from_polar(amp.abs() + 1e-4, phase));
        }
        psi_ret
    }

    /// Construye la onda avanzada $\psi_{\text{adv}}(t)$ desde el atractor o restricción futura
    pub fn build_advanced_wave(&self, target_constraint: &[f64], horizon: f64) -> Vec<ComplexPhasor> {
        let mut psi_adv = Vec::with_capacity(self.dim);
        for j in 0..self.dim {
            let val = if j < target_constraint.len() {
                target_constraint[j]
            } else {
                0.0
            };
            let amp = val.tanh();
            // Onda avanzada viaja hacia atrás en el tiempo: fase negativa
            let phase = -self.frequencies[j] * horizon;
            psi_adv.push(ComplexPhasor::from_polar(amp.abs() + 1e-4, phase));
        }
        psi_adv
    }

    /// Calcula la interferencia de Handshake entre la onda retardada y la avanzada:
    /// Retorna el estado colapsado coherente y la acción de interferencia.
    pub fn compute_handshake_collapse(
        &self,
        psi_ret: &[ComplexPhasor],
        psi_adv: &[ComplexPhasor],
        n_relaxation_steps: usize,
    ) -> (Vec<ComplexPhasor>, f64) {
        let mut state: Vec<ComplexPhasor> = Vec::with_capacity(self.dim);

        // 1. Superposición cuántica inicial: $\psi_{\text{total}} = \frac{1}{\sqrt{2}} (\psi_{\text{ret}} + \psi_{\text{adv}})$
        for j in 0..self.dim {
            let ret = psi_ret.get(j).copied().unwrap_or_else(ComplexPhasor::zero);
            let adv = psi_adv.get(j).copied().unwrap_or_else(ComplexPhasor::zero);
            let sum = ret.add(&adv).scale(std::f64::consts::FRAC_1_SQRT_2);
            state.push(sum);
        }

        // 2. Relajación por Descenso de Energía Libre $\mathcal{F} = \mathcal{U} - T \mathcal{S}$
        // hacia el atractor de interferencia constructiva
        let dt = 0.05;
        let mut total_action = 0.0;

        for _ in 0..n_relaxation_steps {
            let current = state.clone();
            total_action = 0.0;

            for j in 0..self.dim {
                let ret = psi_ret.get(j).copied().unwrap_or_else(ComplexPhasor::zero);
                let adv = psi_adv.get(j).copied().unwrap_or_else(ComplexPhasor::zero);

                // Término de interferencia cruzada constructiva $\Re(\psi_{\text{ret}}^* \cdot \psi_{\text{adv}})$
                let cross = ret.conj().mul(&adv);
                let coherence = cross.re;
                total_action -= coherence;

                // Campo medio de acoplamiento con vecinos
                let mut coupling_field = ComplexPhasor::zero();
                for k in 0..self.dim {
                    let w = self.coupling_matrix[j][k];
                    coupling_field = coupling_field.add(&current[k].scale(w));
                }

                // Fuerza termodinámica de relajación
                let target = coupling_field.scale(1.0 + coherence.max(0.0));
                let diff = target.sub(&current[j]);

                // Normalización de energía y disipación térmica
                let new_re = current[j].re + dt * diff.re;
                let new_im = current[j].im + dt * diff.im;
                let mut updated = ComplexPhasor::new(new_re, new_im);

                // Acotamiento contractivo de norma
                let norm = updated.norm();
                if norm > 1.0 {
                    updated = updated.scale(1.0 / norm);
                }

                state[j] = updated;
            }
        }

        (state, total_action)
    }

    /// Extrae las magnitudes reales y de fase colapsadas como vector de modulación $\in \mathbb{R}^{2d}$
    pub fn extract_real_vector(&self, collapsed_state: &[ComplexPhasor]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.dim * 2);
        for p in collapsed_state {
            out.push(p.re);
            out.push(p.im);
        }
        out
    }

    /// Calcula la coherencia constructiva y peso modulador entre características sensoriales y de prior
    pub fn compute_coherence(&self, sensory_features: &[f64], prior_features: &[f64]) -> (f64, f64) {
        let mut total_coherence = 0.0;
        let n = sensory_features.len().min(prior_features.len()).min(self.dim);
        if n == 0 {
            return (0.5, 0.0);
        }
        for j in 0..n {
            let amp_ret = sensory_features[j].abs().tanh() + 1e-4;
            let amp_adv = prior_features[j].abs().tanh() + 1e-4;
            let phase_diff = self.frequencies[j] * 0.5;
            let coh = amp_ret * amp_adv * phase_diff.cos();
            total_coherence += coh;
        }
        let mean_coh = total_coherence / (n as f64);
        let weight = 1.0 / (1.0 + (-mean_coh * 4.0).exp());
        (weight, -total_coherence)
    }
}
