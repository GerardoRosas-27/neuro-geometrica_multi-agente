//! `AuraAgent` v2: Full Triple-Timescale Cognitive Orchestrator integrating
//! Perception -> DualCorticalCore -> EnsembleWorldModel (Aleatoric/Epistemic) ->
//! HippocampusGraph -> MetacogController.

use aura_cortex::DualCorticalCore;
use aura_hippocampus::{Episode, EpisodicHippocampus, HippocampusGraph};
use aura_math::LatentVec;
use aura_metacog::{MetacogConfig, MetacogController, MetacogReport};
use aura_perception::{PatchEmission, PerceptionEncoder};
use aura_world::EnsembleWorldModel;

pub struct AuraAgent<P: PerceptionEncoder> {
    pub perception: P,
    pub cortex: DualCorticalCore,
    pub world: EnsembleWorldModel,
    pub hippocampus: HippocampusGraph,
    pub metacog: MetacogController,
    pub current_latent: LatentVec,
    pub has_prev: bool,
    pub hipp_enabled: bool,
    pub step_idx: u64,
}

impl<P: PerceptionEncoder> AuraAgent<P> {
    pub fn new(
        perception: P,
        seed: u64,
        fast_enabled: bool,
        hipp_enabled: bool,
        reducible_gating: bool,
        metacog_cfg: MetacogConfig,
    ) -> Self {
        Self {
            perception,
            cortex: DualCorticalCore::new(seed, fast_enabled),
            world: EnsembleWorldModel::new(seed, fast_enabled, reducible_gating),
            hippocampus: HippocampusGraph::new(512, metacog_cfg.hipp_write_threshold),
            metacog: MetacogController::new(metacog_cfg),
            current_latent: LatentVec::zero(),
            has_prev: false,
            hipp_enabled,
            step_idx: 0,
        }
    }

    /// Observe one emitted perceptual patch, predict via WorldModel + Hippocampus recall,
    /// update fast weights & hippocampus gated by reducible epistemic uncertainty.
    /// Returns `(normalized_l1_error, MetacogReport, wrote_hippocampus)`.
    pub fn step_patch(&mut self, em: &PatchEmission) -> (f64, MetacogReport, bool) {
        self.step_idx += 1;
        if !self.has_prev {
            self.current_latent = em.latent;
            self.has_prev = true;
            let _ = self.cortex.step(&em.latent, false);
            let rep = MetacogReport {
                raw_disagreement: 0,
                reducible_epistemic: 0,
                aleatoric_noise: 0,
                surprise_l1: 0,
                learning_progress: 0.0,
                mode: aura_metacog::CognitiveMode::System1Reflex,
                should_store_hippocampus: false,
            };
            return (0.0, rep, false);
        }

        let prev = self.current_latent;
        // 1. If hippocampus is enabled, recall closest past transition for `prev` BEFORE scoring
        //    so one-shot episodic recall reduces surprise immediately on revisiting Regime A!
        let mut hipp_pred_override: Option<LatentVec> = None;
        if self.hipp_enabled && !self.hippocampus.is_empty() {
            let matches = self.hippocampus.recall_with_indices(&prev, 3);
            let prev_norm = prev.norm_l1().max(256);
            if let Some(&(_, dist, ep)) = matches.first() {
                // Scale-invariant L1 similarity gate (relative L1 distance < 0.75)
                if dist * 4 < prev_norm * 3 {
                    self.cortex.write_fast(&ep.key, &ep.next_latent, 1);
                    hipp_pred_override = Some(ep.next_latent);
                }
            }
        }

        // 2. Evaluate world model & decompose aleatoric vs reducible epistemic uncertainty
        let (wm_pred, wm_err, dec) =
            self.world
                .observe_and_decompose(&prev, 0, &em.latent, em.mean_entropy_bits);

        let pred_used = if let Some(hp) = hipp_pred_override {
            let mut blend = LatentVec::zero();
            for d in 0..aura_math::LATENT_DIM {
                blend.data[d] = (((wm_pred.data[d] as i32) + (hp.data[d] as i32) * 2) / 3)
                    .clamp(-127, 127) as i8;
            }
            blend
        } else {
            wm_pred
        };

        let target_ema = self.world.ema_encoder.encode_target(&em.latent);
        let l1_err = target_ema.l1_distance(&pred_used);
        let norm_denom = target_ema.norm_l1().max(256) as f64;
        let norm_err = (l1_err as f64) / norm_denom;

        let rep = self.metacog.diagnose(wm_err, dec);
        let surprise_boost = rep.reducible_epistemic >= self.metacog.cfg.doubt_threshold;
        let _h_out = self.cortex.step(&em.latent, surprise_boost);
        self.cortex.write_fast(&prev, &em.latent, if surprise_boost { 1 } else { 2 });

        // 3. Store salient episode ONLY if metacognition confirms reducible epistemic curiosity
        //    (blocks TV static from polluting Hippocampus!)
        let mut wrote = false;
        if self.hipp_enabled && rep.should_store_hippocampus {
            self.hippocampus.insert(Episode {
                key: prev,
                action_id: 0,
                next_latent: target_ema,
                surprise: rep.reducible_epistemic,
                reward_signal: 1,
                timestamp: self.step_idx,
                payload_id: 0,
                salience: rep.reducible_epistemic.clamp(1, 60000) as u16,
            });
            wrote = true;
        }

        self.current_latent = em.latent;
        (norm_err, rep, wrote)
    }
}
