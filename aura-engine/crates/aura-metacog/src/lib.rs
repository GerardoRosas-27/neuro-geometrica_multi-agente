//! `aura-metacog`: Dual-Process Metacognitive Controller (System 1 Reflex / System 2
//! Depth-2 Expected Free Energy Deliberation / AbstainAndAsk) with Learning-Progress
//! epistemic gating (refuses to trigger deliberation or hippocampal writes on TV static).

use aura_math::LatentVec;
use aura_world::{EnsembleWorldModel, UncertaintyDecomposition};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveMode {
    System1Reflex,
    System2Deliberate,
    AbstainAndAsk,
}

#[derive(Clone, Copy, Debug)]
pub struct MetacogReport {
    pub raw_disagreement: i32,
    pub reducible_epistemic: i32,
    pub aleatoric_noise: i32,
    pub surprise_l1: i32,
    pub learning_progress: f32,
    pub mode: CognitiveMode,
    pub should_store_hippocampus: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct MetacogConfig {
    pub doubt_threshold: i32,
    pub abstain_threshold: i32,
    pub hipp_write_threshold: i32,
    pub system2_enabled: bool,
    pub epi_weight_num: i32,
    pub epi_weight_den: i32,
}

impl Default for MetacogConfig {
    fn default() -> Self {
        Self {
            doubt_threshold: 1400,
            abstain_threshold: 4200,
            hipp_write_threshold: 850,
            system2_enabled: true,
            epi_weight_num: 1,
            epi_weight_den: 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MetacogController {
    pub cfg: MetacogConfig,
    pub s1_count: u64,
    pub s2_count: u64,
    pub abstain_count: u64,
}

impl MetacogController {
    pub fn new(cfg: MetacogConfig) -> Self {
        Self {
            cfg,
            s1_count: 0,
            s2_count: 0,
            abstain_count: 0,
        }
    }

    pub fn reset_counts(&mut self) {
        self.s1_count = 0;
        self.s2_count = 0;
        self.abstain_count = 0;
    }

    /// Classify current cognitive state from observed surprise & uncertainty decomposition.
    pub fn diagnose(&mut self, surprise_l1: i32, dec: UncertaintyDecomposition) -> MetacogReport {
        let eff_epi = dec.reducible_epistemic;
        let mode = if !self.cfg.system2_enabled {
            self.s1_count += 1;
            CognitiveMode::System1Reflex
        } else if eff_epi >= self.cfg.abstain_threshold {
            self.abstain_count += 1;
            CognitiveMode::AbstainAndAsk
        } else if eff_epi >= self.cfg.doubt_threshold {
            self.s2_count += 1;
            CognitiveMode::System2Deliberate
        } else {
            self.s1_count += 1;
            CognitiveMode::System1Reflex
        };

        // Store in hippocampus ONLY when epistemic uncertainty is genuinely reducible
        // (suppresses polluting episodic memory with high-entropy TV static!).
        let should_store = eff_epi >= self.cfg.hipp_write_threshold && dec.learning_progress >= 0.08;

        MetacogReport {
            raw_disagreement: dec.raw_disagreement,
            reducible_epistemic: eff_epi,
            aleatoric_noise: dec.aleatoric_noise,
            surprise_l1,
            learning_progress: dec.learning_progress,
            mode,
            should_store_hippocampus: should_store,
        }
    }

    /// Depth-2 Expected Free Energy rollout over candidate actions using `EnsembleWorldModel`.
    /// Returns `(best_action, best_pred, report)`.
    pub fn arbitrate_action(
        &mut self,
        world: &EnsembleWorldModel,
        state: &LatentVec,
        goal: &LatentVec,
        candidate_actions: &[u16],
        entropy_bits: f32,
        surprise_l1: i32,
    ) -> (u16, LatentVec, MetacogReport) {
        assert!(!candidate_actions.is_empty());
        let reflex = candidate_actions[0];
        let (pred1, dec1) = world.evaluate_uncertainty(state, reflex, entropy_bits);

        if !self.cfg.system2_enabled || dec1.reducible_epistemic < self.cfg.doubt_threshold {
            self.s1_count += 1;
            let rep = MetacogReport {
                raw_disagreement: dec1.raw_disagreement,
                reducible_epistemic: dec1.reducible_epistemic,
                aleatoric_noise: dec1.aleatoric_noise,
                surprise_l1,
                learning_progress: dec1.learning_progress,
                mode: CognitiveMode::System1Reflex,
                should_store_hippocampus: dec1.reducible_epistemic >= self.cfg.hipp_write_threshold,
            };
            return (reflex, pred1, rep);
        }

        // System 2: Depth-2 Expected Free Energy search across candidate actions
        let mut best_act = reflex;
        let mut best_pred = pred1;
        let mut best_efe = i64::MAX;
        let mut best_dec = dec1;

        for &a1 in candidate_actions {
            let (s1, d1) = world.evaluate_uncertainty(state, a1, entropy_bits);
            let mut min_step2 = i64::MAX;
            for &a2 in candidate_actions {
                let (s2, d2) = world.evaluate_uncertainty(&s1, a2, entropy_bits);
                let g2 = s2.l1_distance(goal) as i64;
                let e2 = (d2.reducible_epistemic as i64) * (self.cfg.epi_weight_num as i64)
                    / (self.cfg.epi_weight_den.max(1) as i64);
                min_step2 = min_step2.min(g2 + e2);
            }
            let g1 = s1.l1_distance(goal) as i64;
            let e1 = (d1.reducible_epistemic as i64) * (self.cfg.epi_weight_num as i64)
                / (self.cfg.epi_weight_den.max(1) as i64);
            let efe = g1 + e1 + (min_step2 >> 1);
            if efe < best_efe {
                best_efe = efe;
                best_act = a1;
                best_pred = s1;
                best_dec = d1;
            }
        }

        let mode = if best_dec.reducible_epistemic >= self.cfg.abstain_threshold {
            self.abstain_count += 1;
            CognitiveMode::AbstainAndAsk
        } else {
            self.s2_count += 1;
            CognitiveMode::System2Deliberate
        };

        let rep = MetacogReport {
            raw_disagreement: best_dec.raw_disagreement,
            reducible_epistemic: best_dec.reducible_epistemic,
            aleatoric_noise: best_dec.aleatoric_noise,
            surprise_l1,
            learning_progress: best_dec.learning_progress,
            mode,
            should_store_hippocampus: best_dec.reducible_epistemic >= self.cfg.hipp_write_threshold,
        };
        (best_act, best_pred, rep)
    }
}
