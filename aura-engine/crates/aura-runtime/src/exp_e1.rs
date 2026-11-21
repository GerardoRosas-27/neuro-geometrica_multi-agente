//! Experimento E1 (Etapa 1A): Flujo crudo de bytes SIN TOKENS con:
//! - Fase 1 (0..600): Régimen A estructurado (regla oculta R_A por semilla).
//! - Fase 2 (600..1200): Cambio abrupto a Régimen B (regla oculta R_B) -> recuperación de sorpresa.
//! - Fase 3 (1200..1500): "Estática de TV" (ruido aleatorio uniforme irreducible).
//! - Fase 4A (1500..2100): Retorno a Régimen A -> prueba de recuperación episódica hipocampal.
//! - Fase 4B (2100..2400): Régimen C inédito OOD -> separación de incertidumbre epistémica.
//!
//! Brazos pareados sobre el mismo flujo exacto por semilla:
//! `aura_full`, `no_fast_weights`, `no_hippocampus`, `fixed_patcher`, `no_vicreg`, `persistence`.

use aura_hippocampus::{Episode, EpisodicHippocampus, HippocampusGraph};
use aura_math::{effective_rank_subspace, LatentVec, Xorshift64};
use aura_metacog::{MetacogConfig, MetacogController};
use aura_perception::{EntropyPatcher, FixedPatcher};
use aura_world::EnsembleWorldModel;

#[derive(Clone, Debug)]
pub struct E1Row {
    pub seed: u64,
    pub phase: String,
    pub arm: String,
    pub total_patches: usize,
    pub pred_bytes_per_patch: f64,
    pub high_ent_bytes_per_patch: f64,
    pub compression_ratio: f64,
    pub phase1_norm_err: f64,
    pub phase2_shift_norm_err: f64,
    pub phase2_recovery_steps: f64,
    pub phase3_static_norm_err: f64,
    pub phase3_static_coherence: f64,
    pub phase2_shift_coherence: f64,
    pub phase4_recall_norm_err: f64,
    pub familiar_epistemic: f64,
    pub static_epistemic: f64,
    pub ood_epistemic: f64,
    pub effective_rank: f64,
    pub mean_latent_norm: f64,
    pub eval_ms: f64,
}

#[derive(Clone, Copy)]
struct ByteStreamData {
    bytes: [u8; 2400],
}

fn generate_byte_stream(seed: u64) -> ByteStreamData {
    let mut rng = Xorshift64::from_seed(seed).split_stream(0xE100_5752);
    let mut bytes = [0u8; 2400];

    let keys: [[u8; 2]; 4] = [
        [b'K', b'0' + (rng.next_usize(4) as u8)],
        [b'M', b'4' + (rng.next_usize(4) as u8)],
        [b'Q', b'A' + (rng.next_usize(6) as u8)],
        [b'X', b'G' + (rng.next_usize(6) as u8)],
    ];
    let mut vals_a = [[0u8; 4]; 4];
    let mut vals_b = [[0u8; 4]; 4];
    let mut vals_c = [[0u8; 4]; 4];
    for k in 0..4 {
        let base_a = b'a' + ((k * 5 + rng.next_usize(3)) as u8 % 20);
        let base_b = b'A' + ((k * 5 + 3 + rng.next_usize(3)) as u8 % 20);
        let base_c = b'0' + ((k * 2 + rng.next_usize(2)) as u8 % 8);
        vals_a[k] = [base_a, base_a + 1, base_a + 2, base_a + 3];
        vals_b[k] = [base_b + 3, base_b + 2, base_b + 1, base_b];
        vals_c[k] = [base_c, b'#', base_c + 1, b'@'];
    }

    let emit_frames = |bytes_buf: &mut [u8; 2400],
                       start: usize,
                       end: usize,
                       table: &[[u8; 4]; 4],
                       header: [u8; 4],
                       rng_ref: &mut Xorshift64| {
        let mut pos = start;
        let mut frame_idx = 0usize;
        while pos + 12 <= end {
            let k = (frame_idx + rng_ref.next_usize(2)) & 3;
            let frame = [
                header[0], header[1], header[2], header[3],
                keys[k][0], keys[k][1],
                b'-', b'>',
                table[k][0], table[k][1], table[k][2], table[k][3],
            ];
            bytes_buf[pos..pos + 12].copy_from_slice(&frame);
            pos += 12;
            frame_idx += 1;
        }
    };

    // Fase 1 (0..600): Régimen A
    emit_frames(&mut bytes, 0, 600, &vals_a, *b"SEQ:", &mut rng);
    // Fase 2 (600..1200): Cambio abrupto a Régimen B
    emit_frames(&mut bytes, 600, 1200, &vals_b, *b"SEQ:", &mut rng);
    // Fase 3 (1200..1500): Estática de TV (ruido uniforme aleatorio)
    for b in bytes.iter_mut().take(1500).skip(1200) {
        *b = (rng.next_u64() & 0xFF) as u8;
    }
    // Fase 4A (1500..2100): Retorno a Régimen A
    emit_frames(&mut bytes, 1500, 2100, &vals_a, *b"SEQ:", &mut rng);
    // Fase 4B (2100..2400): Régimen C nunca visto (OOD)
    emit_frames(&mut bytes, 2100, 2400, &vals_c, *b"OOD!", &mut rng);

    ByteStreamData { bytes }
}

#[derive(Clone, Copy)]
struct E1ArmCfg {
    name: &'static str,
    use_fast_weights: bool,
    use_hippocampus: bool,
    use_entropy_patcher: bool,
    use_vicreg: bool,
    is_persistence: bool,
}

fn eval_e1_arm(seed: u64, phase: &str, stream: &ByteStreamData, cfg: E1ArmCfg) -> E1Row {
    let t0 = std::time::Instant::now();
    let mut ent_patcher = EntropyPatcher::new(seed, 1, 6, 420, cfg.use_vicreg);
    let mut fix_patcher = FixedPatcher::new(seed, 4, cfg.use_vicreg);

    let mut wm = EnsembleWorldModel::new(seed, 0, true);
    let mut hip = HippocampusGraph::new(512, 0, 140);
    let mut metacog = MetacogController::new(MetacogConfig::default());

    let mut prev_latent: Option<LatentVec> = None;
    let mut prev_pred: Option<LatentVec> = None;

    let mut total_patches = 0usize;
    let mut p1_late_patch_lens = Vec::new();
    let mut p3_high_ent_patch_lens = Vec::new();

    let mut p1_errs = Vec::new();
    let mut p2_errs = Vec::new();
    let mut p2_cohs = Vec::new();
    let mut p3_errs = Vec::new();
    let mut p3_cohs = Vec::new();
    let mut p4_recall_errs = Vec::new();

    let mut fam_epis = Vec::new();
    let mut static_epis = Vec::new();
    let mut ood_epis = Vec::new();

    let mut collected_latents: Vec<LatentVec> = Vec::with_capacity(600);
    let mut latent_norms = Vec::with_capacity(600);
    let mut recalled_regime_a_loaded = false;

    for byte_idx in 0..2400 {
        let b = stream.bytes[byte_idx];
        let emitted = if cfg.use_entropy_patcher {
            ent_patcher.push_byte_with_meta(b)
        } else {
            fix_patcher.push_byte_with_meta(b)
        };

        let Some((z_obs, meta)) = emitted else {
            continue;
        };
        total_patches += 1;

        if (240..600).contains(&byte_idx) {
            p1_late_patch_lens.push(meta.patch_len as f64);
        } else if (1200..1500).contains(&byte_idx) {
            p3_high_ent_patch_lens.push(meta.patch_len as f64);
        }

        if (600..2100).contains(&byte_idx) {
            collected_latents.push(z_obs);
            latent_norms.push((z_obs.l1_norm() as f64) / 256.0);
        }

        let fast_allowed_now = cfg.use_fast_weights || byte_idx < 600;

        if let (Some(z_prev), Some(z_pred)) = (prev_latent, prev_pred) {
            let raw_err = z_obs.l1_distance(&z_pred) as f64;
            let norm_den = (z_obs.l1_norm().max(64)) as f64;
            let norm_err = raw_err / norm_den;

            let (surprise, coh) = metacog.observe_surprise(&z_obs, &z_pred, 0);

            if cfg.use_entropy_patcher {
                ent_patcher.adapt_encoder_from_prediction(&z_obs, &z_pred);
            } else {
                fix_patcher.adapt_encoder_from_prediction(&z_obs, &z_pred);
            }

            if byte_idx < 600 {
                if byte_idx >= 240 {
                    p1_errs.push(norm_err);
                }
            } else if byte_idx < 1200 {
                p2_errs.push(norm_err);
                p2_cohs.push(coh as f64);
            } else if byte_idx < 1500 {
                p3_errs.push(norm_err);
                p3_cohs.push(coh as f64);
            } else if byte_idx < 2100 {
                p4_recall_errs.push(norm_err);
            }

            if !cfg.is_persistence {
                // Guardar episodios limpios tras el calentamiento del patcher (byte >= 240 en Fase 1).
                if cfg.use_hippocampus
                    && ((240..600).contains(&byte_idx) || (840..1200).contains(&byte_idx))
                {
                    let tag = if byte_idx < 600 { 1u8 } else { 2u8 };
                    hip.insert_episode(Episode {
                        key: z_prev,
                        action_id: 0,
                        next_latent: z_obs,
                        surprise: surprise.max(100),
                        reward_signal: 1,
                        task_tag: tag,
                        replay_count: 0,
                    });
                }

                // Verificación CA1 + pattern completion hipocampal al regresar a un régimen previo:
                // si un episodio almacenado explica la transición observada mejor que el modelo actual,
                // se restauran en memoria rápida las asociaciones recientes de ese régimen.
                if cfg.use_hippocampus
                    && fast_allowed_now
                    && (1500..2100).contains(&byte_idx)
                    && !recalled_regime_a_loaded
                    && raw_err > 1000.0
                {
                    let matches = hip.recall_similar(&z_prev, 6);
                    if let Some(best_ep) = matches
                        .iter()
                        .min_by_key(|ep| ep.next_latent.l1_distance(&z_obs))
                    {
                        let ep_err = best_ep.next_latent.l1_distance(&z_obs) as f64;
                        if ep_err < raw_err * 0.75 {
                            let matched_tag = best_ep.task_tag;
                            // Reinyectar episodios limpios del régimen reconocido en 2 pasadas rápidas.
                            let tag_eps: Vec<Episode> = hip
                                .episodes
                                .iter()
                                .filter(|ep| ep.task_tag == matched_tag)
                                .rev()
                                .take(32)
                                .cloned()
                                .collect();
                            for _pass in 0..2 {
                                for ep in &tag_eps {
                                    wm.observe_transition(
                                        &ep.key,
                                        0,
                                        &ep.next_latent,
                                        1,
                                        false,
                                        false,
                                    );
                                }
                            }
                            recalled_regime_a_loaded = true;
                        }
                    }
                }

                if fast_allowed_now {
                    let lr_shift = if byte_idx < 600 {
                        1
                    } else if (1200..1500).contains(&byte_idx) && coh < 42 {
                        3
                    } else {
                        metacog.fast_lr_shift()
                    };
                    let update_slow = byte_idx < 600;
                    wm.observe_transition(&z_prev, 0, &z_obs, lr_shift, update_slow, false);
                }
            }
        }

        let (next_pred, epi) = if cfg.is_persistence {
            (z_obs, 0)
        } else {
            wm.predict_consensus(&z_obs, 0)
        };
        if (360..600).contains(&byte_idx) {
            fam_epis.push(epi as f64);
        } else if (1320..1500).contains(&byte_idx) {
            static_epis.push(epi as f64);
        } else if byte_idx >= 2100 {
            ood_epis.push(epi as f64);
        }

        prev_latent = Some(z_obs);
        prev_pred = Some(next_pred);
    }

    let mean = |xs: &[f64]| -> f64 {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().sum::<f64>() / (xs.len() as f64)
        }
    };

    let p2_window = p2_errs.len().min(40);
    let p2_shift_err = mean(&p2_errs[..p2_window]);

    let peak_p2 = p2_errs
        .iter()
        .take(6)
        .copied()
        .fold(0.0f64, f64::max)
        .max(1e-6);
    let target_rec = peak_p2 * 0.65;
    let mut rec_steps = p2_window as f64;
    for idx in 2..p2_window {
        let win_avg = (p2_errs[idx] + p2_errs[idx - 1] + p2_errs[idx - 2]) / 3.0;
        if win_avg <= target_rec {
            rec_steps = idx as f64;
            break;
        }
    }

    let p4_win = p4_recall_errs.len().min(35);
    let p4_err = mean(&p4_recall_errs[..p4_win]);

    let pred_bpp = mean(&p1_late_patch_lens);
    let high_ent_bpp = mean(&p3_high_ent_patch_lens);
    let comp_ratio = pred_bpp / high_ent_bpp.max(1e-6);

    let erank = effective_rank_subspace(&collected_latents, 64);

    E1Row {
        seed,
        phase: phase.to_string(),
        arm: cfg.name.to_string(),
        total_patches,
        pred_bytes_per_patch: pred_bpp,
        high_ent_bytes_per_patch: high_ent_bpp,
        compression_ratio: comp_ratio,
        phase1_norm_err: mean(&p1_errs),
        phase2_shift_norm_err: p2_shift_err,
        phase2_recovery_steps: rec_steps,
        phase3_static_norm_err: mean(&p3_errs),
        phase3_static_coherence: mean(&p3_cohs),
        phase2_shift_coherence: mean(&p2_cohs[..p2_cohs.len().min(25)]),
        phase4_recall_norm_err: p4_err,
        familiar_epistemic: mean(&fam_epis),
        static_epistemic: mean(&static_epis),
        ood_epistemic: mean(&ood_epis),
        effective_rank: erank,
        mean_latent_norm: mean(&latent_norms),
        eval_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

pub fn run_e1_for_seed(seed: u64, phase: &str) -> Vec<E1Row> {
    let stream = generate_byte_stream(seed);
    let arms = [
        E1ArmCfg {
            name: "aura_full",
            use_fast_weights: true,
            use_hippocampus: true,
            use_entropy_patcher: true,
            use_vicreg: true,
            is_persistence: false,
        },
        E1ArmCfg {
            name: "no_fast_weights",
            use_fast_weights: false,
            use_hippocampus: false,
            use_entropy_patcher: true,
            use_vicreg: true,
            is_persistence: false,
        },
        E1ArmCfg {
            name: "no_hippocampus",
            use_fast_weights: true,
            use_hippocampus: false,
            use_entropy_patcher: true,
            use_vicreg: true,
            is_persistence: false,
        },
        E1ArmCfg {
            name: "fixed_patcher",
            use_fast_weights: true,
            use_hippocampus: true,
            use_entropy_patcher: false,
            use_vicreg: true,
            is_persistence: false,
        },
        E1ArmCfg {
            name: "no_vicreg",
            use_fast_weights: true,
            use_hippocampus: true,
            use_entropy_patcher: true,
            use_vicreg: false,
            is_persistence: false,
        },
        E1ArmCfg {
            name: "persistence",
            use_fast_weights: false,
            use_hippocampus: false,
            use_entropy_patcher: true,
            use_vicreg: true,
            is_persistence: true,
        },
    ];
    arms.iter()
        .map(|&cfg| eval_e1_arm(seed, phase, &stream, cfg))
        .collect()
}
