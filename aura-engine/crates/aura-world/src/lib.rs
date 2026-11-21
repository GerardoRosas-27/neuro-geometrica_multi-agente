//! `aura-world`: Modelo de Mundo Predictivo en Espacio Latente (estilo JEPA) con ensamble
//! de `K = 4` cabezas independientes, funciones prior aleatorizadas en planos de bits
//! ternarios (para incertidumbre epistémica genuina), memoria rápida NLMS por cabeza
//! y codificador objetivo EMA (`~0.996` por desplazamiento de bits).

use aura_math::{
    decay_sym, LatentVec, TernaryMatrix, Xorshift64, ENSEMBLE_HEADS, LATENT_DIM,
};

/// Dimensión compacta de la clave `(estado, acción)` por cabeza (múltiplo de 64).
pub const KEY_DIM: usize = 64;
pub const MAX_ACTIONS: usize = 8;

/// Contrato del Modelo de Mundo (fiel a la propuesta AURA §4).
pub trait WorldModel {
    /// Predice el siguiente latente con `K = 4` cabezas independientes.
    fn predict_ensemble(
        &self,
        state: &LatentVec,
        action_id: u16,
    ) -> [LatentVec; ENSEMBLE_HEADS];

    /// Calcula la incertidumbre epistémica como desacuerdo L1 medio entre los `C(4,2)=6` pares.
    fn epistemic_disagreement(&self, preds: &[LatentVec; ENSEMBLE_HEADS]) -> i32 {
        let mut total_disp = 0i32;
        for i in 0..ENSEMBLE_HEADS {
            for j in (i + 1)..ENSEMBLE_HEADS {
                total_disp += preds[i].l1_distance(&preds[j]);
            }
        }
        total_disp / 6
    }
}

/// Memoria rápida rectangular `LATENT_DIM x KEY_DIM` (punto fijo Q12, Regla Delta NLMS).
#[derive(Clone, Debug)]
pub struct FastAssocHead {
    pub weights: Vec<i32>, // LATENT_DIM * KEY_DIM en Q12
    pub decay_shift: u8,
}

impl FastAssocHead {
    pub fn new(decay_shift: u8) -> Self {
        Self {
            weights: vec![0i32; LATENT_DIM * KEY_DIM],
            decay_shift,
        }
    }

    pub fn reset(&mut self) {
        self.weights.fill(0);
    }

    #[inline]
    pub fn query(&self, key: &[i8; KEY_DIM], out: &mut [i32; LATENT_DIM]) {
        for (r, slot) in out.iter_mut().enumerate() {
            let row = &self.weights[r * KEY_DIM..(r + 1) * KEY_DIM];
            let mut acc = 0i64;
            for c in 0..KEY_DIM {
                acc += (row[c] as i64) * (key[c] as i64);
            }
            *slot = (acc >> 12) as i32;
        }
    }

    /// Regla Delta Normalizada (NLMS entera) con contractividad garantizada para cualquier `||key||`.
    pub fn update_residual(
        &mut self,
        key: &[i8; KEY_DIM],
        target_residual: &[i32; LATENT_DIM],
        lr_shift: u8,
    ) {
        let mut norm_sq: i64 = 0;
        for &k in key {
            let kv = k as i64;
            norm_sq += kv * kv;
        }
        if norm_sq == 0 {
            return;
        }
        let k_log2 = (63 - norm_sq.leading_zeros()) as i32;
        let net_shift = k_log2 - 12 + (lr_shift.max(1) as i32);
        let decay = self.decay_shift;

        for r in 0..LATENT_DIM {
            let row = &mut self.weights[r * KEY_DIM..(r + 1) * KEY_DIM];
            let mut acc = 0i64;
            for c in 0..KEY_DIM {
                acc += (row[c] as i64) * (key[c] as i64);
            }
            let pred = (acc >> 12) as i32;
            let err = target_residual[r].saturating_sub(pred) as i64;
            for c in 0..KEY_DIM {
                let decayed = if decay > 0 {
                    decay_sym(row[c], decay)
                } else {
                    row[c]
                };
                let prod = err * (key[c] as i64);
                let shifted = if net_shift >= 0 {
                    prod >> (net_shift.min(48) as u32)
                } else {
                    prod.saturating_mul(1i64 << ((-net_shift).min(12) as u32))
                };
                let delta = shifted.clamp(-500_000, 500_000) as i32;
                row[c] = decayed.saturating_add(delta).clamp(-8_000_000, 8_000_000);
            }
        }
    }
}

/// Cabeza individual del ensamble `K = 4`:
/// - `key_proj`: proyección dispersa ternaria 256 -> 64 (0-mul) específica de la cabeza.
/// - `prior_net`: función prior aleatoria fija en planos de bits ternarios (Osband-style)
///   que genera desacuerdo genuino en regiones no visitadas y se cancela al aprender.
/// - `slow_net`: pesos lentos consolidados ternarios (`256 x 64`, 0-mul).
/// - `fast_net`: memoria rápida online NLMS (`256 x 64`, punto fijo Q12).
#[derive(Clone)]
struct WorldHead {
    key_proj: TernaryMatrix,  // 64 x 256
    prior_net: TernaryMatrix, // 256 x 64 (fija, 0-mul)
    slow_net: TernaryMatrix,  // 256 x 64 (consolidable en sueño, 0-mul)
    fast_net: FastAssocHead,  // 256 x 64 (online NLMS)
}

/// Codificador objetivo EMA estilo JEPA (`theta_bar <- (1 - 1/256) theta_bar + (1/256) theta`).
#[derive(Clone)]
pub struct EmaTargetEncoder {
    /// Ganancias suavizadas por dimensión en Q8 (inicialmente 256 = 1.0).
    pub target_scale_q8: [i32; LATENT_DIM],
    pub target_bias_q8: [i32; LATENT_DIM],
    pub ema_shift: u8, // 8 => tasa 1/256 ≈ 0.0039 (análogo a 0.996 / 0.004)
}

impl EmaTargetEncoder {
    pub fn new(ema_shift: u8) -> Self {
        Self {
            target_scale_q8: [256; LATENT_DIM],
            target_bias_q8: [0; LATENT_DIM],
            ema_shift: ema_shift.clamp(1, 10),
        }
    }

    /// Proyecta un latente observado a través del codificador objetivo suavizado por EMA.
    #[inline]
    pub fn encode_target(&self, obs: &LatentVec) -> LatentVec {
        let mut data = [0i8; LATENT_DIM];
        for i in 0..LATENT_DIM {
            let v = (((obs.data[i] as i32) * self.target_scale_q8[i]) + self.target_bias_q8[i]) >> 8;
            data[i] = v.clamp(-127, 127) as i8;
        }
        LatentVec { data }
    }

    /// Actualiza parámetros EMA hacia la señal observada (estabilización JEPA).
    pub fn update_ema(&mut self, obs: &LatentVec) {
        let s = self.ema_shift;
        for i in 0..LATENT_DIM {
            let tgt_bias = (obs.data[i] as i32) << 4;
            self.target_bias_q8[i] += (tgt_bias - self.target_bias_q8[i]) >> s;
        }
    }
}

/// Modelo de Mundo completo con `K = 4` cabezas, condicionamiento por acción y EMA objetivo.
#[derive(Clone)]
pub struct EnsembleWorldModel {
    heads: [WorldHead; ENSEMBLE_HEADS],
    action_sign_masks: [[i8; KEY_DIM]; MAX_ACTIONS],
    pub ema_encoder: EmaTargetEncoder,
    pub use_fast_weights: bool,
    pub predict_residual: bool,
    pub prior_shift: u8, // Escala del prior aleatorio (controla magnitud de desacuerdo OOD)
    rng: Xorshift64,
}

impl EnsembleWorldModel {
    pub fn new(seed: u64, decay_shift: u8, use_fast_weights: bool) -> Self {
        let mut rng = Xorshift64::from_seed(seed).split_stream(0x574F_524C);
        let heads = std::array::from_fn(|_| WorldHead {
            key_proj: TernaryMatrix::new_random_sparse(KEY_DIM, LATENT_DIM, 22, &mut rng),
            prior_net: TernaryMatrix::new_random_sparse(LATENT_DIM, KEY_DIM, 18, &mut rng),
            slow_net: TernaryMatrix::new_zeros(LATENT_DIM, KEY_DIM),
            fast_net: FastAssocHead::new(decay_shift),
        });

        let mut action_sign_masks = [[1i8; KEY_DIM]; MAX_ACTIONS];
        for mask in &mut action_sign_masks {
            for slot in mask.iter_mut() {
                *slot = if (rng.next_u64() & 1) == 0 { 1 } else { -1 };
            }
        }

        Self {
            heads,
            action_sign_masks,
            ema_encoder: EmaTargetEncoder::new(8),
            use_fast_weights,
            predict_residual: false,
            prior_shift: 4,
            rng,
        }
    }

    pub fn with_residual_mode(mut self, predict_residual: bool) -> Self {
        self.predict_residual = predict_residual;
        self
    }

    pub fn reset_fast_weights(&mut self) {
        for h in &mut self.heads {
            h.fast_net.reset();
        }
    }

    /// Construye la clave compacta `64D` de `(state, action_id)` para la cabeza `head_idx` (0-mul).
    /// Cuando `predict_residual = true`, los primeros 8 canales codifican una firma ortogonal
    /// de la acción (ley global de desplazamiento) y los 56 restantes la interacción estado×acción
    /// (obstáculos locales y detección de zonas OOD).
    #[inline]
    fn encode_key(&self, head_idx: usize, state: &LatentVec, action_id: u16) -> [i8; KEY_DIM] {
        let mut raw = [0i32; KEY_DIM];
        self.heads[head_idx]
            .key_proj
            .project_zero_mul(&state.data, &mut raw);
        let act_idx = (action_id as usize) % MAX_ACTIONS;
        let mask = &self.action_sign_masks[act_idx];
        let mut key = [0i8; KEY_DIM];
        let state_start = if self.predict_residual {
            for i in 0..8 {
                key[i] = if i == act_idx { 64 } else { 0 };
            }
            8
        } else {
            0
        };
        for i in state_start..KEY_DIM {
            let shifted = raw[i] >> 4;
            let signed = if mask[i] > 0 { shifted } else { -shifted };
            key[i] = signed.clamp(-127, 127) as i8;
        }
        key
    }

    /// Predicción interna de una sola cabeza `k` en `i32` (antes de cuantizar a `i8`).
    fn predict_head_raw(
        &self,
        head_idx: usize,
        state: &LatentVec,
        key: &[i8; KEY_DIM],
        out: &mut [i32; LATENT_DIM],
    ) {
        let h = &self.heads[head_idx];
        let mut prior_out = [0i32; LATENT_DIM];
        let mut slow_out = [0i32; LATENT_DIM];
        h.prior_net.project_zero_mul(key, &mut prior_out);
        h.slow_net.project_zero_mul(key, &mut slow_out);

        let mut fast_out = [0i32; LATENT_DIM];
        if self.use_fast_weights {
            h.fast_net.query(key, &mut fast_out);
        }

        for i in 0..LATENT_DIM {
            let base_skip = if self.predict_residual {
                state.data[i] as i32
            } else {
                0
            };
            let fast_term = if self.use_fast_weights { fast_out[i] } else { 0 };
            out[i] =
                base_skip + (prior_out[i] >> self.prior_shift) + (slow_out[i] >> 3) + fast_term;
        }
    }

    /// Observa una transición real `(state, action_id) -> next_state` y actualiza las cabezas.
    pub fn observe_transition(
        &mut self,
        state: &LatentVec,
        action_id: u16,
        next_state: &LatentVec,
        lr_shift: u8,
        update_slow: bool,
        bootstrap: bool,
    ) {
        let target = if self.predict_residual {
            *next_state
        } else {
            let t = self.ema_encoder.encode_target(next_state);
            self.ema_encoder.update_ema(next_state);
            t
        };

        let mut prior_out = [0i32; LATENT_DIM];
        let mut slow_out = [0i32; LATENT_DIM];
        let mut target_res = [0i32; LATENT_DIM];

        for k in 0..ENSEMBLE_HEADS {
            if bootstrap && (self.rng.next_u64() & 3) == (k as u64) {
                continue;
            }
            let key = self.encode_key(k, state, action_id);
            self.heads[k].prior_net.project_zero_mul(&key, &mut prior_out);

            if update_slow {
                self.heads[k].slow_net.project_zero_mul(&key, &mut slow_out);
                for r in 0..LATENT_DIM {
                    let skip = if self.predict_residual {
                        state.data[r] as i32
                    } else {
                        0
                    };
                    let pred_base =
                        skip + (prior_out[r] >> self.prior_shift) + (slow_out[r] >> 3);
                    let err = (target.data[r] as i32) - pred_base;
                    if err.abs() < 4 {
                        continue;
                    }
                    let err_sign = err.signum() as i16;
                    for c in 0..KEY_DIM {
                        let kc = key[c];
                        if kc.abs() >= 12 {
                            let delta = err_sign * (kc.signum() as i16) * 16;
                            self.heads[k].slow_net.consolidate_sleep_weight(r, c, delta);
                        }
                    }
                }
            }

            if self.use_fast_weights {
                self.heads[k].slow_net.project_zero_mul(&key, &mut slow_out);
                for r in 0..LATENT_DIM {
                    let skip = if self.predict_residual {
                        state.data[r] as i32
                    } else {
                        0
                    };
                    let base = skip + (prior_out[r] >> self.prior_shift) + (slow_out[r] >> 3);
                    target_res[r] = (target.data[r] as i32) - base;
                }
                self.heads[k]
                    .fast_net
                    .update_residual(&key, &target_res, lr_shift);
            }
        }
    }

    /// Consenso central (mediana entera L1) de las `K = 4` predicciones.
    pub fn predict_consensus(&self, state: &LatentVec, action_id: u16) -> (LatentVec, i32) {
        let preds = self.predict_ensemble(state, action_id);
        let epi = self.epistemic_disagreement(&preds);
        let consensus = LatentVec::median_of_four(&preds);
        (consensus, epi)
    }
}

impl WorldModel for EnsembleWorldModel {
    fn predict_ensemble(
        &self,
        state: &LatentVec,
        action_id: u16,
    ) -> [LatentVec; ENSEMBLE_HEADS] {
        let mut out = [LatentVec::zero(); ENSEMBLE_HEADS];
        let mut buf = [0i32; LATENT_DIM];
        for k in 0..ENSEMBLE_HEADS {
            let key = self.encode_key(k, state, action_id);
            self.predict_head_raw(k, state, &key, &mut buf);
            out[k] = LatentVec::from_i32_shifted(&buf, 0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_latent(rng: &mut Xorshift64) -> LatentVec {
        let mut v = LatentVec::zero();
        for x in &mut v.data {
            *x = (rng.next_signed_byte() >> 1).clamp(-127, 127) as i8;
        }
        v
    }

    #[test]
    fn epistemic_uncertainty_drops_on_seen_and_stays_high_on_unseen() {
        let mut rng = Xorshift64::from_seed(0xA400);
        let mut wm = EnsembleWorldModel::new(0xA400, 0, true);
        let s_seen = random_latent(&mut rng);
        let t_seen = random_latent(&mut rng);
        let s_unseen = random_latent(&mut rng);

        let (_, epi_before) = wm.predict_consensus(&s_seen, 0);
        for _ in 0..8 {
            wm.observe_transition(&s_seen, 0, &t_seen, 1, false, false);
        }
        let (pred_seen, epi_after) = wm.predict_consensus(&s_seen, 0);
        let (_, epi_unseen) = wm.predict_consensus(&s_unseen, 0);

        assert!(
            epi_after * 3 < epi_before,
            "epistemic disagreement did not drop after learning: before={epi_before}, after={epi_after}"
        );
        assert!(
            epi_unseen > epi_after * 3,
            "unseen state should have much higher epistemic disagreement ({epi_unseen}) than seen ({epi_after})"
        );
        let err = pred_seen.l1_distance(&t_seen);
        assert!(err < 1200, "prediction error too large after adaptation: {err}");
    }

    #[test]
    fn aleatoric_tv_static_keeps_low_epistemic_vs_unseen_state() {
        let mut rng = Xorshift64::from_seed(0xA401);
        let mut wm = EnsembleWorldModel::new(0xA401, 0, true);
        let s_noisy = random_latent(&mut rng);
        let s_unseen = random_latent(&mut rng);

        // Exponer `s_noisy` a ruido simétrico de media cero ("estática de TV").
        for _ in 0..40 {
            let noise_target = random_latent(&mut rng);
            wm.observe_transition(&s_noisy, 1, &noise_target, 3, false, false);
        }
        let (_, epi_noisy) = wm.predict_consensus(&s_noisy, 1);
        let (_, epi_unseen) = wm.predict_consensus(&s_unseen, 1);
        assert!(
            epi_noisy < epi_unseen,
            "seen aleatoric noise ({epi_noisy}) should have lower epistemic uncertainty than unseen OOD state ({epi_unseen})"
        );
    }
}
