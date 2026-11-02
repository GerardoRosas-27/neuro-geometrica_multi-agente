//! Oscilaciones Neurales (Acoplamiento Fase-Amplitud Theta-Gamma) y Sistemas Neuromoduladores
//! Subcorticales (Dopamina, Acetilcolina, Norepinefrina, Serotonina).
//!
//! Fundamentación neurocientífica:
//! 1. **Buzsáki (2006)**, *Rhythms of the Brain* (Oxford Univ. Press); **Lisman & Jensen (2013)**,
//!    *Neuron*, 77(6), 1002-1016 (*The Theta-Gamma Neural Code*):
//!    Cada ciclo Theta ($4\text{-}8\text{ Hz}$) anida múltiples subciclos Gamma ($30\text{-}80\text{ Hz}$)
//!    generados por el bucle excitatorio-inhibitorio Piramidal-PV+ (PING).
//! 2. **Hasselmo, Bodelón & Wyble (2002)**, *Neural Computation*, 14(4), 793-817:
//!    El ritmo Theta segrega temporalmente la **codificación sensorial** (fase de valle: predominio
//!    *bottom-up* entorrinal/talámico y alta plasticidad LTP) de la **recuperación/predicción**
//!    (fase de pico: predominio *top-down* cortical y completado de patrones en CA3).
//! 3. **Doya (2002)**, *Neural Networks*, 15(4-6), 495-506; **Yu & Dayan (2005)**, *Neuron*, 46(4), 681-692;
//!    **Schultz, Dayan & Montague (1997)**, *Science*, 275, 1593-1599:
//!    - **Dopamina (DA, VTA/SNc)**: Error de predicción de recompensa/desempeño ($\delta_{\text{RPE}}$).
//!    - **Acetilcolina (ACh, Prosencéfalo Basal / NBM)**: Incertidumbre esperada; amplifica la
//!      ganancia sensorial *bottom-up* y activa interneuronas VIP+.
//!    - **Norepinefrina (NE, Locus Coeruleus)**: Sorpresa inesperada (cambio de contexto / volatilidad);
//!      ejecuta un *network reset* (Bouret & Sara, 2005) y acelera la tasa de aprendizaje.
//!    - **Serotonina (5-HT, Núcleo del Rafe)**: Control de estabilidad homeostática y escala temporal.

use std::f64::consts::TAU;

/// Oscilador acoplado Theta-Gamma (Buzsáki, 2006; Hasselmo et al., 2002).
#[derive(Clone, Debug)]
pub struct ThetaGammaOscillator {
    /// Fase instantánea Theta $\phi_\theta \in [0, 2\pi)$.
    pub theta_phase: f64,
    /// Fase instantánea Gamma $\phi_\gamma \in [0, 2\pi)$.
    pub gamma_phase: f64,
    /// Número de subciclos Gamma por cada ciclo Theta (típicamente 4 a 7, Lisman & Jensen, 2013).
    pub gamma_per_theta: usize,
    /// Incremento de fase Theta por subpaso.
    pub d_theta: f64,
}

impl Default for ThetaGammaOscillator {
    fn default() -> Self {
        Self::new(5)
    }
}

impl ThetaGammaOscillator {
    pub fn new(gamma_per_theta: usize) -> Self {
        let gpt = gamma_per_theta.max(2);
        Self {
            theta_phase: 0.0,
            gamma_phase: 0.0,
            gamma_per_theta: gpt,
            d_theta: TAU / (gpt as f64),
        }
    }

    /// Avanza un subciclo Gamma dentro del ciclo Theta actual.
    pub fn step_gamma_subcycle(&mut self) {
        self.theta_phase = (self.theta_phase + self.d_theta).rem_euclid(TAU);
        self.gamma_phase =
            (self.gamma_phase + self.d_theta * (self.gamma_per_theta as f64)).rem_euclid(TAU);
    }

    /// Peso de la fase de codificación sensorial *bottom-up* (Hasselmo et al., 2002):
    /// Máximo en la primera mitad del ciclo Theta.
    pub fn encoding_gate(&self) -> f64 {
        0.55 + 0.45 * self.theta_phase.cos()
    }

    /// Peso de la fase de predicción *top-down* y recuperación hipocampal CA3:
    /// Complementario a la fase de codificación.
    pub fn retrieval_gate(&self) -> f64 {
        0.55 - 0.45 * self.theta_phase.cos()
    }

    /// Envolvente de sincronía Gamma (acoplamiento fase-amplitud Theta-Gamma).
    pub fn gamma_synchrony_envelope(&self) -> f64 {
        let theta_mod = 0.7 + 0.3 * self.theta_phase.sin().abs();
        let gamma_wave = 0.5 * (1.0 + self.gamma_phase.cos());
        theta_mod * (0.6 + 0.4 * gamma_wave)
    }
}

/// Estado de los cuatro sistemas neuromoduladores ascendentes del tronco encefálico y prosencéfalo basal.
#[derive(Clone, Debug)]
pub struct NeuromodulatorySystem {
    /// Tono fásico de Dopamina (VTA/SNc, $\text{DA} \in [0.1, 2.5]$).
    pub dopamine: f64,
    /// Nivel de Acetilcolina (Núcleo Basal de Meynert, $\text{ACh} \in [0.1, 2.0]$).
    pub acetylcholine: f64,
    /// Nivel fásico de Norepinefrina (Locus Coeruleus, $\text{NE} \in [0.05, 2.5]$).
    pub norepinephrine: f64,
    /// Tono de Serotonina (Núcleo del Rafe Dorsal, $\text{5-HT} \in [0.2, 1.5]$).
    pub serotonin: f64,
    /// Media móvil de error/sorpresa esperada ($\bar{S}$, incertidumbre esperada de Yu & Dayan, 2005).
    pub expected_surprise: f64,
    /// Media móvil de desempeño/recompensa ($\bar{R}$) para calcular el error de predicción dopaminérgico.
    pub baseline_reward: f64,
    /// Indicador de disparo de *network reset* noradrenérgico ante cambio abrupto de régimen.
    pub context_switch_detected: bool,
}

impl Default for NeuromodulatorySystem {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuromodulatorySystem {
    pub fn new() -> Self {
        Self {
            dopamine: 1.0,
            acetylcholine: 0.5,
            norepinephrine: 0.2,
            serotonin: 0.8,
            expected_surprise: 0.25,
            baseline_reward: 0.0,
            context_switch_detected: false,
        }
    }

    /// Actualiza los cuatro neuromoduladores a partir del error de predicción instantáneo ($S_t$)
    /// y la señal de recompensa/éxito ($R_t = -S_t$ en aprendizaje auto-supervisado o recompensa externa).
    pub fn update(&mut self, instantaneous_surprise: f64, external_reward: Option<f64>) {
        let s = instantaneous_surprise.max(0.0);

        // 1. Razón de sorpresa inesperada vs esperada (Yu & Dayan, 2005):
        // Si la sorpresa actual supera ampliamente la incertidumbre esperada, el Locus Coeruleus
        // emite una descarga fásica de Norepinefrina (NE) señalizando cambio de contexto.
        let surprise_ratio = s / (self.expected_surprise + 1e-4);
        self.context_switch_detected = surprise_ratio > 2.2 && s > 0.15;

        if self.context_switch_detected {
            self.norepinephrine = (0.4 + 0.65 * (surprise_ratio - 1.0)).clamp(0.2, 2.5);
        } else {
            self.norepinephrine =
                (0.82 * self.norepinephrine + 0.18 * (0.15 + 0.25 * s)).clamp(0.05, 2.0);
        }

        // 2. Actualizar incertidumbre esperada (ACh en Prosencéfalo Basal / NBM)
        let alpha_s = if self.context_switch_detected {
            0.35
        } else {
            0.08
        };
        self.expected_surprise = (1.0 - alpha_s) * self.expected_surprise + alpha_s * s;
        self.acetylcholine = (0.30 + 1.20 * self.expected_surprise.tanh()).clamp(0.15, 2.0);

        // 3. Error de Predicción de Recompensa Dopaminérgico (VTA, Schultz et al., 1997)
        let reward = external_reward.unwrap_or(-s);
        let rpe = reward - self.baseline_reward;
        self.baseline_reward = 0.90 * self.baseline_reward + 0.10 * reward;
        // La dopamina tónica + fásica codifica tanto el RPE como la saliencia motivacional
        self.dopamine = (1.0 + 0.65 * rpe.tanh() + 0.25 * self.norepinephrine).clamp(0.20, 2.5);

        // 4. Serotonina (5-HT): inversamente acoplada a la volatilidad noradrenérgica
        self.serotonin = (1.1 - 0.35 * self.norepinephrine).clamp(0.25, 1.4);
    }

    /// Factor modulador global de 3 factores ($M(t)$) que gobierna la magnitud de la plasticidad
    /// sináptica cortical (combinación sinérgica de DA, ACh y NE, Gerstner et al., 2018).
    pub fn third_factor_plasticity_gain(&self) -> f64 {
        let gain = self.dopamine * (0.55 + 0.30 * self.acetylcholine + 0.35 * self.norepinephrine);
        gain.clamp(0.15, 3.0)
    }
}
