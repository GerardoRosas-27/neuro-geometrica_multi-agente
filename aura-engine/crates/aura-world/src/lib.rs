//! `aura-world`: JEPA Latent World Model (K=4 Ensemble with Randomized Ternary Priors,
//! EMA Target Encoder 0.996, and Heteroscedastic Aleatoric + Learning-Progress Reducible
//! Epistemic Decomposition solving the TV-static curiosity trap in Bottleneck 5).

use aura_math::{
    FastWeightMemory, LatentVec, TernaryMatrix, Xorshift64, ENSEMBLE_HEADS, LATENT_DIM,
};

/// Full uncertainty decomposition produced at each step.
#[derive(Clone, Copy, Debug)]
pub struct UncertaintyDecomposition {
    /// Raw pairwise L1 disagreement across K=4 ensemble heads (v1 metric).
    pub raw_disagreement: i32,
    /// Bottleneck 5.1 fix: Reducible epistemic uncertainty (curiosity / info-gain),
    /// gated by structural compressibility and online learning progress.
    pub reducible_epistemic: i32,
    /// Estimated irreducible aleatoric noise (high on TV static, low on structured regimes).
    pub aleatoric_noise: i32,
    /// Online learning progress score in [0.0, 1.0].
    pub learning_progress: f32,
}

/// Core WorldModel trait from the proposal.
pub trait WorldModel {
    fn predict_ensemble(&self, state: &LatentVec, action_id: u16) -> [LatentVec; ENSEMBLE_HEADS];

    fn epistemic_disagreement(&self, preds: &[LatentVec; ENSEMBLE_HEADS]) -> i32 {
        let mut total = 0i32;
        for i in 0..ENSEMBLE_HEADS {
            for j in (i + 1)..ENSEMBLE_HEADS {
                total += preds[i].l1_distance(&preds[j]);
            }
        }
        total / 6
    }

    fn consensus_prediction(&self, preds: &[LatentVec; ENSEMBLE_HEADS]) -> LatentVec {
        let mut out = LatentVec::zero();
        for d in 0..LATENT_DIM {
            let sum = (preds[0].data[d] as i32)
                + (preds[1].data[d] as i32)
                + (preds[2].data[d] as i32)
                + (preds[3].data[d] as i32);
            out.data[d] = (sum >> 2).clamp(-127, 127) as i8;
        }
        out
    }
}

/// EMA Target Encoder (JEPA: theta_bar <- 0.996 * theta_bar + 0.004 * theta).
#[derive(Clone, Debug)]
pub struct EmaTargetEncoder {
    pub weights: Vec<f32>, // 256 diagonal + mixing gain updated by EMA
    pub ema_momentum: f32,
    pub online_gain: Vec<f32>,
}

impl EmaTargetEncoder {
    pub fn new(ema_momentum: f32) -> Self {
        Self {
            weights: vec![1.0f32; LATENT_DIM],
            ema_momentum,
            online_gain: vec![1.0f32; LATENT_DIM],
        }
    }

    pub fn encode_target(&self, z: &LatentVec) -> LatentVec {
        let mut out = LatentVec::zero();
        for d in 0..LATENT_DIM {
            out.data[d] = ((z.data[d] as f32) * self.weights[d])
                .round()
                .clamp(-127.0, 127.0) as i8;
        }
        out
    }

    pub fn update_ema(&mut self, observed: &LatentVec) {
        let m = self.ema_momentum;
        for d in 0..LATENT_DIM {
            let mag = (observed.data[d] as f32).abs();
            // Gentle online normalization gain targeting ~48 Q7 amplitude
            let target_g = if mag > 4.0 {
                (48.0 / mag).clamp(0.65, 1.45)
            } else {
                1.0
            };
            self.online_gain[d] = 0.98 * self.online_gain[d] + 0.02 * target_g;
            self.weights[d] = m * self.weights[d] + (1.0 - m) * self.online_gain[d];
        }
    }
}

/// One independent world-model ensemble head:
/// fixed Randomized Prior (`TernaryMatrix`, zero-mul) + trainable NLMS associative residual.
#[derive(Clone, Debug)]
pub struct EnsembleHead {
    pub prior: TernaryMatrix,
    pub learnable: FastWeightMemory,
}

impl EnsembleHead {
    pub fn new(seed: u64, decay_shift: u8) -> Self {
        let mut rng = Xorshift64::new(seed);
        let prior = TernaryMatrix::from_random_density(LATENT_DIM, LATENT_DIM, 0.25, &mut rng);
        Self {
            prior,
            learnable: FastWeightMemory::new(LATENT_DIM, decay_shift),
        }
    }

    fn condition_input(state: &LatentVec, action_id: u16) -> [i8; LATENT_DIM] {
        let mut key = state.data;
        if action_id > 0 {
            let rot = ((action_id as usize) * 19) % LATENT_DIM;
            key.rotate_left(rot);
            let tag = (((action_id as i32) * 37) % 120 - 60) as i8;
            for k in (0..LATENT_DIM).step_by(16) {
                key[k] = key[k].saturating_add(tag / 2);
            }
        }
        key
    }

    pub fn predict(&self, state: &LatentVec, action_id: u16) -> LatentVec {
        let key = Self::condition_input(state, action_id);
        let mut p_out = [0i32; LATENT_DIM];
        let mut l_out = [0i32; LATENT_DIM];
        self.prior.project_lut4_simd(&key, &mut p_out);
        self.learnable.query(&key, &mut l_out);

        let mut out = LatentVec::zero();
        for d in 0..LATENT_DIM {
            // Prior scaled to ~Q7/4 + learnable residual cancels prior & predicts target
            let val = (p_out[d] >> 4) + (l_out[d] >> 4);
            out.data[d] = val.clamp(-127, 127) as i8;
        }
        out
    }

    pub fn update_toward(&mut self, state: &LatentVec, action_id: u16, target: &LatentVec, lr_shift: u8) {
        let key = Self::condition_input(state, action_id);
        let mut p_out = [0i32; LATENT_DIM];
        self.prior.project_lut4_simd(&key, &mut p_out);

        let mut residual_target = [0i32; LATENT_DIM];
        for d in 0..LATENT_DIM {
            // Want (p_out[d] >> 4) + (l_out[d] >> 4) == target[d]
            // => l_out[d] == (target[d] - (p_out[d] >> 4)) << 4
            let desired = (target.data[d] as i32) - (p_out[d] >> 4);
            residual_target[d] = desired << 4;
        }
        self.learnable.update_online_nlms(&key, &residual_target, lr_shift);
    }

    pub fn reset(&mut self) {
        self.learnable.reset();
    }
}

/// JEPA K=4 Ensemble World Model with Aleatoric/Epistemic Learning-Progress Decomposition.
#[derive(Clone, Debug)]
pub struct EnsembleWorldModel {
    pub heads: [EnsembleHead; ENSEMBLE_HEADS],
    pub ema_encoder: EmaTargetEncoder,
    pub fast_enabled: bool,
    pub lr_shift: u8,
    /// Enable Bottleneck 5.1 fix (heteroscedastic aleatoric + learning-progress gating).
    pub reducible_gating_enabled: bool,

    // Online causal statistics for aleatoric vs epistemic separation (ZERO label leakage!)
    ema_entropy_bits: f32,
    ema_error_l1: f32,
    ema_post_update_ratio: f32,
    ema_error_sign: [f32; 32],
    ema_coherence: f32,
}

impl EnsembleWorldModel {
    pub fn new(seed: u64, fast_enabled: bool, reducible_gating_enabled: bool) -> Self {
        let heads = std::array::from_fn(|k| {
            let decay = 8 + (k as u8); // Slightly diverse decay rates across heads
            EnsembleHead::new(seed ^ (0xA0B0_C0D0_0000_0001u64 + (k as u64) * 9973), decay)
        });
        Self {
            heads,
            ema_encoder: EmaTargetEncoder::new(0.996),
            fast_enabled,
            lr_shift: 1,
            reducible_gating_enabled,
            ema_entropy_bits: 3.5,
            ema_error_l1: 2000.0,
            ema_post_update_ratio: 0.45,
            ema_error_sign: [0.0; 32],
            ema_coherence: 0.25,
        }
    }

    pub fn reset(&mut self) {
        for h in &mut self.heads {
            h.reset();
        }
        self.ema_entropy_bits = 3.5;
        self.ema_error_l1 = 2000.0;
        self.ema_post_update_ratio = 0.45;
        self.ema_error_sign.fill(0.0);
        self.ema_coherence = 0.25;
    }

    /// Observe transition `(prev_state, action_id -> next_state)` with causal byte entropy
    /// `patch_entropy_bits` and compute full UncertaintyDecomposition + online update.
    pub fn observe_and_decompose(
        &mut self,
        prev_state: &LatentVec,
        action_id: u16,
        next_observed: &LatentVec,
        patch_entropy_bits: f32,
    ) -> (LatentVec, i32, UncertaintyDecomposition) {
        let preds = self.predict_ensemble(prev_state, action_id);
        let consensus = self.consensus_prediction(&preds);
        let target = self.ema_encoder.encode_target(next_observed);

        let err_before = target.l1_distance(&consensus);
        let raw_dis = self.epistemic_disagreement(&preds);

        // Update directional error coherence (high on structured shift, ~0 on white noise)
        let mut coh_acc = 0.0f32;
        for d in 0..32 {
            let sgn = ((target.data[d * 8] as i32) - (consensus.data[d * 8] as i32)).signum() as f32;
            self.ema_error_sign[d] = 0.80 * self.ema_error_sign[d] + 0.20 * sgn;
            coh_acc += self.ema_error_sign[d].abs();
        }
        self.ema_coherence = 0.85 * self.ema_coherence + 0.15 * (coh_acc / 32.0);

        // Smooth causal byte entropy EMA
        self.ema_entropy_bits = 0.80 * self.ema_entropy_bits + 0.20 * patch_entropy_bits;
        self.ema_error_l1 = 0.85 * self.ema_error_l1 + 0.15 * (err_before as f32);

        // Structural compressibility factor from causal byte entropy:
        // Pure uniform TV static has H ~ 7.3..8.0 bits -> struct_gate ~ 0.05
        // Structured OOD or shift has H ~ 1.5..5.6 bits -> struct_gate ~ 0.55..1.0
        let struct_gate = ((7.0 - self.ema_entropy_bits) / 2.6).clamp(0.05, 1.0);

        // Proposal §3.3: "Incertidumbre aleatoria (ruido del mundo): se ignora para el aprendizaje."
        // When reducible_gating_enabled is true: skip polluting world-model heads on pure TV static
        // (struct_gate < 0.18), keeping the exact same structured learning rate as baseline.
        if self.fast_enabled {
            let skip_aleatoric = self.reducible_gating_enabled && struct_gate < 0.18;
            if !skip_aleatoric {
                for h in &mut self.heads {
                    h.update_toward(prev_state, action_id, &target, self.lr_shift);
                }
            }
        }
        self.ema_encoder.update_ema(next_observed);

        // Measure post-update contraction ratio (Learning Progress / Compression Progress)
        let preds_after = self.predict_ensemble(prev_state, action_id);
        let consensus_after = self.consensus_prediction(&preds_after);
        let err_after = target.l1_distance(&consensus_after);
        let step_ratio = (err_after as f32) / ((err_before as f32).max(1.0));
        self.ema_post_update_ratio = 0.85 * self.ema_post_update_ratio + 0.15 * step_ratio;

        // Learning-progress + directional coherence boost (separates structured novelty from noise)
        let learn_prog = (1.0 - self.ema_post_update_ratio).clamp(0.0, 1.0);
        let coh_boost = (0.45 + 1.2 * self.ema_coherence).clamp(0.35, 1.15);
        let gate = if self.reducible_gating_enabled {
            (struct_gate * struct_gate * (0.25 + 0.75 * learn_prog) * coh_boost).clamp(0.02, 1.0)
        } else {
            1.0
        };

        let reducible_epi = ((raw_dis as f32) * gate).round() as i32;
        let aleatoric = (self.ema_error_l1 * (1.0 - 0.85 * struct_gate)).round() as i32;

        let decomp = UncertaintyDecomposition {
            raw_disagreement: raw_dis,
            reducible_epistemic: reducible_epi,
            aleatoric_noise: aleatoric,
            learning_progress: learn_prog * struct_gate,
        };
        (consensus, err_before, decomp)
    }

    /// Evaluate reducible epistemic uncertainty without mutating weights (used by System 2).
    pub fn evaluate_uncertainty(
        &self,
        state: &LatentVec,
        action_id: u16,
        entropy_bits: f32,
    ) -> (LatentVec, UncertaintyDecomposition) {
        let preds = self.predict_ensemble(state, action_id);
        let consensus = self.consensus_prediction(&preds);
        let raw_dis = self.epistemic_disagreement(&preds);
        let struct_gate = ((7.0 - entropy_bits) / 2.6).clamp(0.05, 1.0);
        let gate = if self.reducible_gating_enabled {
            (struct_gate * struct_gate).clamp(0.02, 1.0)
        } else {
            1.0
        };
        let decomp = UncertaintyDecomposition {
            raw_disagreement: raw_dis,
            reducible_epistemic: ((raw_dis as f32) * gate).round() as i32,
            aleatoric_noise: (self.ema_error_l1 * (1.0 - 0.85 * struct_gate)).round() as i32,
            learning_progress: (1.0 - self.ema_post_update_ratio).clamp(0.0, 1.0) * struct_gate,
        };
        (consensus, decomp)
    }
}

impl WorldModel for EnsembleWorldModel {
    fn predict_ensemble(&self, state: &LatentVec, action_id: u16) -> [LatentVec; ENSEMBLE_HEADS] {
        [
            self.heads[0].predict(state, action_id),
            self.heads[1].predict(state, action_id),
            self.heads[2].predict(state, action_id),
            self.heads[3].predict(state, action_id),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_model_reducible_epistemic_suppresses_tv_static_vs_structured_ood() {
        let mut wm = EnsembleWorldModel::new(123, true, true);
        let mut rng = Xorshift64::new(456);

        // Simulate 40 steps of high-entropy TV static (entropy ~ 7.7 bits)
        let mut static_epi_sum = 0i64;
        let mut prev = LatentVec::zero();
        for _ in 0..40 {
            let mut next = LatentVec::zero();
            for d in 0..LATENT_DIM {
                next.data[d] = ((rng.next_u32() % 200) as i32 - 100) as i8;
            }
            let (_, _, dec) = wm.observe_and_decompose(&prev, 0, &next, 7.8);
            static_epi_sum += dec.reducible_epistemic as i64;
            prev = next;
        }
        let static_mean = (static_epi_sum as f64) / 40.0;

        // Now simulate 40 steps of unseen structured OOD rule (entropy ~ 3.2 bits)
        let mut ood_epi_sum = 0i64;
        let mut rule_states = Vec::new();
        for k in 0..4 {
            let mut v = LatentVec::zero();
            for d in 0..LATENT_DIM {
                v.data[d] = (((k * 53 + d * 17) % 160) as i32 - 80) as i8;
            }
            rule_states.push(v);
        }
        for step in 0..40 {
            let s = rule_states[step % 4];
            let next = rule_states[(step + 1) % 4];
            let (_, _, dec) = wm.observe_and_decompose(&s, 0, &next, 3.0);
            if step >= 8 {
                ood_epi_sum += dec.reducible_epistemic as i64;
            }
        }
        let ood_mean = (ood_epi_sum as f64) / 32.0;
        assert!(
            ood_mean > static_mean * 2.0,
            "Expected structured OOD reducible epistemic ({ood_mean:.1}) >> TV static ({static_mean:.1})"
        );
    }
}
