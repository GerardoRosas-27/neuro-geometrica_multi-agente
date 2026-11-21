//! `aura-metacog`: Controlador Metacognitivo Dual (Sistema 1 reflejo / Sistema 2 deliberativo
//! con rollouts mentales de Energía Libre Esperada / `AbstainAndAsk`) acoplado al bucle
//! neuromodulador (Noradrenalina, Acetilcolina, Dopamina) y detector de volatilidad direccional
//! que separa ruido aleatorio ("estática de TV") de cambios reales de ley física.

use aura_hippocampus::EpisodicHippocampus;
use aura_math::{LatentVec, LATENT_DIM};
use aura_world::{EnsembleWorldModel, WorldModel};

pub const ABSTAIN_ACTION: u16 = u16::MAX;
const PROBE_DIMS: usize = 32;

/// Modo cognitivo seleccionado en el ciclo actual (fiel a la propuesta AURA §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveMode {
    System1Reflex,     // Baja incertidumbre -> respuesta directa en 1 pase
    System2Deliberate, // Alta incertidumbre -> recall hipocampal + rollout mental H=2
    AbstainAndAsk,     // Incertidumbre irreductible tras deliberar -> abstenerse / preguntar
}

/// Diagnóstico metacognitivo completo emitido en cada ciclo.
#[derive(Clone, Copy, Debug)]
pub struct MetacogReport {
    pub epistemic_uncertainty: i32,
    pub surprise_l1: i32,
    pub directional_coherence_pct: i32,
    pub noradrenaline: u8,
    pub acetylcholine: i8,
    pub dopamine: i8,
    pub rollouts_executed: usize,
    pub mode: CognitiveMode,
}

#[derive(Clone, Copy, Debug)]
pub struct MetacogConfig {
    pub doubt_threshold: i32,
    pub abstain_threshold: i32,
    pub ne_deliberate_thresh: u8,
    pub epi_weight_q2: i32, // Peso de incertidumbre epistémica en EFE (en cuartos, p.ej. 4 = 1.0)
    pub enable_system2: bool,
    pub enable_abstain: bool,
    pub enable_hippocampal_recall: bool,
}

impl Default for MetacogConfig {
    fn default() -> Self {
        Self {
            doubt_threshold: 1400,
            abstain_threshold: 2800,
            ne_deliberate_thresh: 6,
            epi_weight_q2: 3,
            enable_system2: true,
            enable_abstain: true,
            enable_hippocampal_recall: true,
        }
    }
}

/// Controlador metacognitivo de segundo orden.
#[derive(Clone, Debug)]
pub struct MetacogController {
    pub config: MetacogConfig,
    pub noradrenaline: u8,
    pub acetylcholine: i8,
    pub dopamine: i8,
    pub ema_surprise: i32,
    pub prev_surprise: i32,
    signed_err_ema: [i32; PROBE_DIMS],
    abs_err_ema: [i32; PROBE_DIMS],
    steps_seen: usize,
}

impl MetacogController {
    pub fn new(config: MetacogConfig) -> Self {
        Self {
            config,
            noradrenaline: 2,
            acetylcholine: 0,
            dopamine: 0,
            ema_surprise: 0,
            prev_surprise: 0,
            signed_err_ema: [0; PROBE_DIMS],
            abs_err_ema: [0; PROBE_DIMS],
            steps_seen: 0,
        }
    }

    pub fn reset_episode_buffers(&mut self) {
        self.signed_err_ema.fill(0);
        self.abs_err_ema.fill(0);
    }

    /// Observa la sorpresa real `|obs - pred|_1` y actualiza NE, ACh, DA separando
    /// volatilidad direccional (cambio de regla) de ruido aleatorio de media cero ("estática TV").
    /// Devuelve `(surprise_l1, directional_coherence_pct)`.
    pub fn observe_surprise(
        &mut self,
        observed: &LatentVec,
        predicted: &LatentVec,
        reward_signal: i32,
    ) -> (i32, i32) {
        let surprise = observed.l1_distance(predicted);
        let stride = LATENT_DIM / PROBE_DIMS;
        let mut sum_signed_abs = 0i64;
        let mut sum_abs = 0i64;

        for p in 0..PROBE_DIMS {
            let idx = p * stride;
            let err = (observed.data[idx] as i32 - predicted.data[idx] as i32) << 4;
            self.signed_err_ema[p] += (err - self.signed_err_ema[p]) >> 2;
            self.abs_err_ema[p] += (err.abs() - self.abs_err_ema[p]) >> 2;
            sum_signed_abs += self.signed_err_ema[p].abs() as i64;
            sum_abs += self.abs_err_ema[p] as i64;
        }
        let coherence_pct = if sum_abs > 16 {
            ((sum_signed_abs * 100) / sum_abs).clamp(0, 100) as i32
        } else {
            0
        };

        if self.steps_seen == 0 {
            self.ema_surprise = surprise;
            self.prev_surprise = surprise;
            self.steps_seen = 1;
            return (surprise, coherence_pct);
        }
        self.steps_seen += 1;

        // Dopamina: reducción de Energía Libre o señal de recompensa externa.
        let delta_fe = self.prev_surprise - surprise;
        self.dopamine = if reward_signal > 0 || delta_fe > 120 {
            1
        } else if reward_signal < 0 || delta_fe < -120 {
            -1
        } else {
            0
        };

        // Noradrenalina: se dispara ante sorpresa por encima de EMA *Y* coherencia direccional
        // (evita dispararse falsamente por ruido blanco de media cero / estática de TV).
        let surprise_spike = surprise - self.ema_surprise;
        let volatility_trigger =
            surprise_spike > (self.ema_surprise >> 2).max(160) && coherence_pct >= 42;
        if volatility_trigger {
            self.noradrenaline = (self.noradrenaline + 4).min(15);
        } else {
            self.noradrenaline = self.noradrenaline.saturating_sub(1);
        }

        // Acetilcolina: alta cuando hay volatilidad (confiar en sentidos), baja en estabilidad.
        if volatility_trigger {
            self.acetylcholine = (self.acetylcholine + 2).min(4);
        } else if surprise < self.ema_surprise {
            self.acetylcholine = (self.acetylcholine - 1).max(-3);
        }

        self.ema_surprise += (surprise - self.ema_surprise) >> 3;
        self.prev_surprise = surprise;
        (surprise, coherence_pct)
    }

    /// Tasa de escritura rápida (`lr_shift` para `FastWeightMemory`) modulada por Noradrenalina.
    /// Menor `lr_shift` = mayor tasa de aprendizaje instantáneo (1 => ~0.5, 2 => ~0.25, 3 => ~0.125).
    #[inline]
    pub fn fast_lr_shift(&self) -> u8 {
        if self.noradrenaline >= 6 {
            1
        } else if self.noradrenaline >= 3 {
            2
        } else {
            3
        }
    }

    /// Arbitraje Sistema 1 / Sistema 2 / AbstainAndAsk sobre un modelo de mundo `EnsembleWorldModel`
    /// y un hipocampo episódico `H`, buscando minimizar Energía Libre Esperada hacia `goal_latent`.
    #[allow(clippy::too_many_arguments)]
    pub fn arbitrate_and_plan<H: EpisodicHippocampus>(
        &mut self,
        current_latent: &LatentVec,
        goal_latent: &LatentVec,
        candidate_actions: &[u16],
        reflex_action: u16,
        world_model: &mut EnsembleWorldModel,
        hippocampus: &H,
        last_surprise: i32,
        coherence_pct: i32,
    ) -> (u16, LatentVec, MetacogReport) {
        assert!(!candidate_actions.is_empty());

        // 1. Evaluar acción refleja (Sistema 1) en 1 solo pase del modelo de mundo.
        let reflex_preds = world_model.predict_ensemble(current_latent, reflex_action);
        let reflex_epi = world_model.epistemic_disagreement(&reflex_preds);
        let reflex_mean = LatentVec::median_of_four(&reflex_preds);

        // Si Sistema 2 está desactivado (ablación) o la incertidumbre epistémica y volatilidad
        // son bajas -> ruta rápida Sistema 1.
        let needs_system2 = self.config.enable_system2
            && (reflex_epi >= self.config.doubt_threshold
                || self.noradrenaline >= self.config.ne_deliberate_thresh);

        if !needs_system2 {
            return (
                reflex_action,
                reflex_mean,
                MetacogReport {
                    epistemic_uncertainty: reflex_epi,
                    surprise_l1: last_surprise,
                    directional_coherence_pct: coherence_pct,
                    noradrenaline: self.noradrenaline,
                    acetylcholine: self.acetylcholine,
                    dopamine: self.dopamine,
                    rollouts_executed: 1,
                    mode: CognitiveMode::System1Reflex,
                },
            );
        }

        // 2. Modo Sistema 2: consulta hipocampal + inyección rápida en memoria de trabajo.
        if self.config.enable_hippocampal_recall && world_model.use_fast_weights {
            let recalled = hippocampus.recall_similar(current_latent, 4);
            for ep in recalled {
                // Solo inyectar recuerdos suficientemente cercanos para no contaminar otro contexto.
                if ep.key.l1_distance(current_latent) < 4000 {
                    world_model.observe_transition(
                        &ep.key,
                        ep.action_id,
                        &ep.next_latent,
                        2,
                        false,
                        false,
                    );
                }
            }
        }

        // 3. Rollout mental en árbol de profundidad H = 2 buscando mínima Energía Libre Esperada:
        //    EFE(a1) = min_{a2} [ d_goal(z1) + d_goal(z2) + lambda_epi * (epi1 + epi2 / 2) ]
        let mut best_action = candidate_actions[0];
        let mut best_efe = i64::MAX;
        let mut best_epi = i32::MAX;
        let mut best_pred = reflex_mean;
        let mut rollouts = 1usize; // Ya ejecutamos 1 en Sistema 1

        for &a1 in candidate_actions {
            let p1 = world_model.predict_ensemble(current_latent, a1);
            let epi1 = world_model.epistemic_disagreement(&p1);
            let z1 = LatentVec::median_of_four(&p1);
            let d1 = z1.l1_distance(goal_latent) as i64;
            rollouts += 1;

            // Segundo paso de rollout mental (lookahead H=2) para detectar atajos/callejones.
            let mut best_h2 = i64::MAX;
            for &a2 in candidate_actions {
                let p2 = world_model.predict_ensemble(&z1, a2);
                let epi2 = world_model.epistemic_disagreement(&p2);
                let z2 = LatentVec::median_of_four(&p2);
                let d2 = z2.l1_distance(goal_latent) as i64;
                let epi_term = (((epi1 + (epi2 >> 1)) as i64) * (self.config.epi_weight_q2 as i64)) >> 2;
                let efe = d1 + d2 + epi_term;
                if efe < best_h2 {
                    best_h2 = efe;
                }
                rollouts += 1;
            }

            if best_h2 < best_efe {
                best_efe = best_h2;
                best_action = a1;
                best_epi = epi1;
                best_pred = z1;
            }
        }

        // 4. Si tras deliberar incluso la mejor acción tiene incertidumbre epistémica irreductible,
        //    emitir `AbstainAndAsk` en lugar de alucinar una acción ciega.
        if self.config.enable_abstain && best_epi >= self.config.abstain_threshold {
            return (
                ABSTAIN_ACTION,
                best_pred,
                MetacogReport {
                    epistemic_uncertainty: best_epi,
                    surprise_l1: last_surprise,
                    directional_coherence_pct: coherence_pct,
                    noradrenaline: self.noradrenaline,
                    acetylcholine: self.acetylcholine,
                    dopamine: self.dopamine,
                    rollouts_executed: rollouts,
                    mode: CognitiveMode::AbstainAndAsk,
                },
            );
        }

        (
            best_action,
            best_pred,
            MetacogReport {
                epistemic_uncertainty: best_epi,
                surprise_l1: last_surprise,
                directional_coherence_pct: coherence_pct,
                noradrenaline: self.noradrenaline,
                acetylcholine: self.acetylcholine,
                dopamine: self.dopamine,
                rollouts_executed: rollouts,
                mode: CognitiveMode::System2Deliberate,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aura_math::Xorshift64;

    #[test]
    fn directional_coherence_distinguishes_law_shift_from_tv_static() {
        let mut rng = Xorshift64::from_seed(77);
        let mut mc_static = MetacogController::new(MetacogConfig::default());
        let mut mc_shift = MetacogController::new(MetacogConfig::default());

        let pred = LatentVec::zero();
        // Calentar ambos con baja sorpresa.
        for _ in 0..10 {
            let obs = LatentVec::zero();
            mc_static.observe_surprise(&obs, &pred, 0);
            mc_shift.observe_surprise(&obs, &pred, 0);
        }

        // Estática de TV: signos aleatorios simétricos de media cero.
        let mut coh_static_sum = 0i32;
        for _ in 0..20 {
            let mut obs = LatentVec::zero();
            for x in &mut obs.data {
                *x = if (rng.next_u64() & 1) == 0 { 40 } else { -40 };
            }
            let (_, coh) = mc_static.observe_surprise(&obs, &pred, 0);
            coh_static_sum += coh;
        }

        // Cambio de ley determinista: desplazamiento consistente de signo.
        let mut coh_shift_sum = 0i32;
        for _ in 0..20 {
            let mut obs = LatentVec::zero();
            for (i, x) in obs.data.iter_mut().enumerate() {
                *x = if (i & 1) == 0 { 40 } else { -40 };
            }
            let (_, coh) = mc_shift.observe_surprise(&obs, &pred, 0);
            coh_shift_sum += coh;
        }

        assert!(
            coh_shift_sum > coh_static_sum * 2,
            "expected higher coherence on law shift ({coh_shift_sum}) than TV static ({coh_static_sum})"
        );
        assert!(
            mc_shift.noradrenaline > mc_static.noradrenaline,
            "expected NE spike on shift ({}) > static ({})",
            mc_shift.noradrenaline,
            mc_static.noradrenaline
        );
    }
}
