//! Orquestador `AuraAgent` unificando los 5 subsistemas de la arquitectura AURA:
//! Percepción sin tokens -> Núcleo Cortical Dual -> Modelo de Mundo K=4 ->
//! Hipocampo Episódico -> Control Metacognitivo Dual (Sistema 1 / Sistema 2 / Abstain).

use aura_cortex::DualCorticalCore;
use aura_hippocampus::{Episode, EpisodicHippocampus, HippocampusGraph};
use aura_math::LatentVec;
use aura_metacog::{MetacogConfig, MetacogController, MetacogReport};
use aura_perception::PerceptionEncoder;
use aura_world::EnsembleWorldModel;

pub struct AuraAgent<P: PerceptionEncoder> {
    pub perception: P,
    pub cortex: DualCorticalCore,
    pub world_model: EnsembleWorldModel,
    pub hippocampus: HippocampusGraph,
    pub metacog: MetacogController,
    pub current_latent: LatentVec,
    pub last_prediction: LatentVec,
    pub has_prev: bool,
}

impl<P: PerceptionEncoder> AuraAgent<P> {
    pub fn new(
        perception: P,
        seed: u64,
        decay_shift: u8,
        use_fast_weights: bool,
        metacog_cfg: MetacogConfig,
    ) -> Self {
        Self {
            perception,
            cortex: DualCorticalCore::new(seed, decay_shift),
            world_model: EnsembleWorldModel::new(seed, decay_shift, use_fast_weights),
            hippocampus: HippocampusGraph::new(512, 1200, 120),
            metacog: MetacogController::new(metacog_cfg),
            current_latent: LatentVec::zero(),
            last_prediction: LatentVec::zero(),
            has_prev: false,
        }
    }

    /// Ejecuta un ciclo cognitivo completo ante la llegada de un parche perceptual `observed_latent`.
    pub fn cognitive_step(
        &mut self,
        observed_latent: LatentVec,
        goal_latent: &LatentVec,
        candidate_actions: &[u16],
        reflex_action: u16,
        prev_action: Option<u16>,
        reward_signal: i32,
        task_tag: u8,
    ) -> (u16, MetacogReport) {
        // 1. Medir sorpresa observada y actualizar bucle neuromodulador (NE, ACh, DA).
        let (surprise, coh) = if self.has_prev {
            self.metacog
                .observe_surprise(&observed_latent, &self.last_prediction, reward_signal)
        } else {
            (0, 0)
        };

        // 2. Actualizar núcleo cortical dual (SSM selectivo + pesos rápidos).
        let prev_latent = self.current_latent;
        let _cortical = self.cortex.step_cortex(
            &observed_latent,
            self.metacog.acetylcholine,
            self.world_model.use_fast_weights,
        );

        // 3. Aprendizaje online del Modelo de Mundo y almacenamiento hipocampal si hay transición previa.
        if let (true, Some(act)) = (self.has_prev, prev_action) {
            let lr_shift = self.metacog.fast_lr_shift();
            if self.world_model.use_fast_weights {
                self.cortex.adapt_fast(&prev_latent, &observed_latent, lr_shift);
            }
            self.world_model
                .observe_transition(&prev_latent, act, &observed_latent, lr_shift, false, true);

            if self.metacog.config.enable_hippocampal_recall {
                self.hippocampus.store_if_salient(Episode {
                    key: prev_latent,
                    action_id: act,
                    next_latent: observed_latent,
                    surprise,
                    reward_signal: reward_signal.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    task_tag,
                    replay_count: 0,
                });
            }
        }

        self.current_latent = observed_latent;
        self.has_prev = true;

        // 4. Arbitraje Metacognitivo Dual: Sistema 1 / Sistema 2 (EFE H=2) / AbstainAndAsk.
        let (chosen_action, pred_next, report) = self.metacog.arbitrate_and_plan(
            &self.current_latent,
            goal_latent,
            candidate_actions,
            reflex_action,
            &mut self.world_model,
            &self.hippocampus,
            surprise,
            coh,
        );
        self.last_prediction = pred_next;
        (chosen_action, report)
    }
}
