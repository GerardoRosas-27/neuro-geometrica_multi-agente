//! Experimento E2 (Etapa 1B): Mundo Activo de Física Discreta 2D (6x6) codificado en tramas
//! sensoriales crudas, con cambio abrupto de ley física (`L_1 -> L_2`) y sondas de incertidumbre
//! OOD (trampas ambiguas) que evalúan el arbitraje metacognitivo Sistema 1 / Sistema 2 / Abstain.
//!
//! Brazos pareados: `aura_full`, `system1_only`, `no_fast_weights`, `random_control`.

use aura_hippocampus::{Episode, EpisodicHippocampus, HippocampusGraph};
use aura_math::{LatentVec, TernaryMatrix, Xorshift64, LATENT_DIM};
use aura_metacog::{CognitiveMode, MetacogConfig, MetacogController, ABSTAIN_ACTION};
use aura_world::EnsembleWorldModel;

const GRID: usize = 6;
const N_ACTIONS: u16 = 4;
const CANDIDATE_ACTIONS: [u16; 4] = [0, 1, 2, 3];

#[derive(Clone, Debug)]
pub struct E2Row {
    pub seed: u64,
    pub phase: String,
    pub arm: String,
    pub phase1_return: f64,
    pub phase1_success: f64,
    pub phase2_shift_return: f64,
    pub phase2_shift_success: f64,
    pub phase3_ood_return: f64,
    pub combined_p2_p3_return: f64,
    pub abstain_rate_trap: f64,
    pub abstain_rate_familiar: f64,
    pub system1_frac_familiar: f64,
    pub mean_rollouts_per_step: f64,
    pub eval_ms: f64,
}

#[derive(Clone)]
struct World2D {
    obstacles: [[bool; GRID]; GRID],
    goal: (usize, usize),
    law2_perm: [usize; 4],
    ood_proj: TernaryMatrix,
    ood_safe_action: [u16; 8],
}

impl World2D {
    fn new(seed: u64) -> Self {
        let mut rng = Xorshift64::from_seed(seed).split_stream(0xE200_2D2D);
        let mut obstacles = [[false; GRID]; GRID];
        let gx = 4 + rng.next_usize(2);
        let gy = 4 + rng.next_usize(2);
        let goal = (gx, gy);

        // Colocar 2 obstáculos interiores sin bloquear bordes ni desconectar el mapa.
        let candidates = [(2, 2), (2, 3), (3, 2), (1, 3), (3, 1)];
        let mut placed = 0;
        for _ in 0..16 {
            if placed >= 2 {
                break;
            }
            let (ox, oy) = candidates[rng.next_usize(candidates.len())];
            if !obstacles[ox][oy] && (ox, oy) != goal {
                obstacles[ox][oy] = true;
                placed += 1;
            }
        }

        // Permutación sin puntos fijos (derangement) para la Ley física 2.
        let perms = [[2, 3, 1, 0], [3, 2, 0, 1], [1, 0, 3, 2]];
        let law2_perm = perms[rng.next_usize(perms.len())];

        let ood_proj = TernaryMatrix::new_random_sparse(LATENT_DIM, LATENT_DIM, 22, &mut rng);
        let mut ood_safe_action = [0u16; 8];
        for slot in &mut ood_safe_action {
            *slot = rng.next_usize(4) as u16;
        }

        Self {
            obstacles,
            goal,
            law2_perm,
            ood_proj,
            ood_safe_action,
        }
    }

    fn apply_action(&self, pos: (usize, usize), action: u16, shifted_law: bool) -> ((usize, usize), bool) {
        let dir = if shifted_law {
            self.law2_perm[(action as usize) % 4]
        } else {
            (action as usize) % 4
        };
        let (dx, dy): (i32, i32) = match dir {
            0 => (0, -1), // N
            1 => (0, 1),  // S
            2 => (-1, 0), // W
            _ => (1, 0),  // E
        };
        let nx = pos.0 as i32 + dx;
        let ny = pos.1 as i32 + dy;
        if nx < 0 || ny < 0 || nx >= GRID as i32 || ny >= GRID as i32 {
            return (pos, true);
        }
        let npos = (nx as usize, ny as usize);
        if self.obstacles[npos.0][npos.1] {
            return (pos, true);
        }
        (npos, false)
    }

    /// Política refleja greedy (Sistema 1) aprendida bajo Ley 1.
    fn reflex_action_law1(&self, pos: (usize, usize)) -> u16 {
        let mut best_a = 0u16;
        let mut best_d = i32::MAX;
        for a in CANDIDATE_ACTIONS {
            let (np, hit) = self.apply_action(pos, a, false);
            let d = (np.0 as i32 - self.goal.0 as i32).abs()
                + (np.1 as i32 - self.goal.1 as i32).abs()
                + if hit { 10 } else { 0 };
            if d < best_d {
                best_d = d;
                best_a = a;
            }
        }
        best_a
    }

    /// Codifica `(pos, zone_id)` en un `LatentVec` 256D:
    /// - Canales 0..192: geometría espacial continua `(x, y)` (alinea distancia L1 con meta).
    /// - Canales 192..224: sensores locales de pared N/S/W/E.
    /// - Canales 224..256: firma de zona sensorial (0 en el mundo conocido; firma ortogonal
    ///   proyectada 0-mul en zonas trampa OOD nunca vistas).
    fn encode_state(&self, pos: (usize, usize), zone_id: u8) -> LatentVec {
        let mut data = [0i8; LATENT_DIM];
        let x_val = ((pos.0 as i32) - 2) * 24;
        let y_val = ((pos.1 as i32) - 2) * 24;
        for i in 0..96 {
            data[i] = x_val.clamp(-120, 120) as i8;
            data[96 + i] = y_val.clamp(-120, 120) as i8;
        }
        // Sensores locales de muro adyacente (observables por el agente).
        let (_, wall_n) = self.apply_action(pos, 0, false);
        let (_, wall_s) = self.apply_action(pos, 1, false);
        let (_, wall_w) = self.apply_action(pos, 2, false);
        let (_, wall_e) = self.apply_action(pos, 3, false);
        for i in 0..8 {
            data[192 + i] = if wall_n { 32 } else { -32 };
            data[200 + i] = if wall_s { 32 } else { -32 };
            data[208 + i] = if wall_w { 32 } else { -32 };
            data[216 + i] = if wall_e { 32 } else { -32 };
        }

        if zone_id == 0 {
            for i in 224..256 {
                data[i] = if (i & 1) == 0 { 16 } else { -16 };
            }
        } else {
            let mut raw = [0i8; LATENT_DIM];
            let z = zone_id as usize;
            for i in 0usize..LATENT_DIM {
                let code = (i.wrapping_mul(37) ^ z.wrapping_mul(101)) & 3;
                raw[i] = match code {
                    0 => 110,
                    1 => -110,
                    2 => 80,
                    _ => -80,
                };
            }
            let mut proj = [0i32; LATENT_DIM];
            self.ood_proj.project_zero_mul(&raw, &mut proj);
            for i in 192..256 {
                data[i] = (proj[i] >> 2).clamp(-127, 127) as i8;
            }
        }
        LatentVec { data }
    }
}

#[derive(Clone, Copy)]
struct E2ArmCfg {
    name: &'static str,
    enable_system2: bool,
    enable_abstain: bool,
    enable_fast_weights_shift: bool,
    is_random: bool,
}

fn eval_e2_arm(seed: u64, phase: &str, world: &World2D, cfg: E2ArmCfg) -> E2Row {
    let t0 = std::time::Instant::now();
    let mut rng = Xorshift64::from_seed(seed).split_stream(0xE2AA_0001);
    let mut wm = EnsembleWorldModel::new(seed, 0, true).with_residual_mode(true);
    let mut hip = HippocampusGraph::new(512, 800, 150);

    // Calibración metacognitiva (DEV): Sistema 1 en estados familiares (epi baja), Sistema 2 en
    // cambio de ley (NE alta o epi media), AbstainAndAsk en zonas OOD nunca vistas (epi muy alta).
    let mc_cfg = MetacogConfig {
        doubt_threshold: 1900,
        abstain_threshold: 2150,
        ne_deliberate_thresh: 5,
        epi_weight_q2: 2,
        enable_system2: cfg.enable_system2,
        enable_abstain: cfg.enable_abstain,
        enable_hippocampal_recall: false,
    };
    let mut metacog = MetacogController::new(mc_cfg);
    let goal_latent = world.encode_state(world.goal, 0);

    // Preentrenar el modelo de mundo bajo Ley 1 sobre celdas accesibles.
    let mut valid_cells = Vec::new();
    for x in 0..GRID {
        for y in 0..GRID {
            if !world.obstacles[x][y] && (x, y) != world.goal {
                valid_cells.push((x, y));
            }
        }
    }
    for _epoch in 0..4 {
        for &pos in &valid_cells {
            let s = world.encode_state(pos, 0);
            for a in CANDIDATE_ACTIONS {
                let (np, _) = world.apply_action(pos, a, false);
                let ns = world.encode_state(np, 0);
                wm.observe_transition(&s, a, &ns, 1, true, false);
            }
        }
    }

    let starts: [(usize, usize); 8] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 1),
        (0, 3),
        (3, 0),
    ];

    // FASE 1: Evaluación en Ley 1 familiar (24 episodios).
    let mut p1_ret_sum = 0.0f64;
    let mut p1_succ = 0usize;
    let mut s1_fam_steps = 0usize;
    let mut tot_fam_steps = 0usize;
    let mut rollouts_sum = 0usize;
    let mut rollouts_n = 0usize;

    for ep in 0..24 {
        let mut pos = starts[ep % starts.len()];
        if world.obstacles[pos.0][pos.1] {
            pos = (0, 0);
        }
        metacog.reset_episode_buffers();
        metacog.noradrenaline = 1;
        let mut ep_ret = 0.0f64;
        let mut reached = false;
        let mut last_pred = world.encode_state(pos, 0);

        for _step in 0..16 {
            let s = world.encode_state(pos, 0);
            let (surp, coh) = metacog.observe_surprise(&s, &last_pred, 0);
            let reflex = world.reflex_action_law1(pos);

            let (act, pred_next, rep) = if cfg.is_random {
                let a = rng.next_usize(N_ACTIONS as usize) as u16;
                let (p, e) = wm.predict_consensus(&s, a);
                (
                    a,
                    p,
                    aura_metacog::MetacogReport {
                        epistemic_uncertainty: e,
                        surprise_l1: surp,
                        directional_coherence_pct: coh,
                        noradrenaline: 0,
                        acetylcholine: 0,
                        dopamine: 0,
                        rollouts_executed: 1,
                        mode: CognitiveMode::System1Reflex,
                    },
                )
            } else {
                metacog.arbitrate_and_plan(
                    &s,
                    &goal_latent,
                    &CANDIDATE_ACTIONS,
                    reflex,
                    &mut wm,
                    &hip,
                    surp,
                    coh,
                )
            };

            tot_fam_steps += 1;
            if rep.mode == CognitiveMode::System1Reflex {
                s1_fam_steps += 1;
            }
            rollouts_sum += rep.rollouts_executed;
            rollouts_n += 1;

            let chosen = if act == ABSTAIN_ACTION { reflex } else { act };
            let (npos, hit) = world.apply_action(pos, chosen, false);
            let r = if npos == world.goal {
                reached = true;
                20.0
            } else if hit {
                -2.0
            } else {
                -1.0
            };
            ep_ret += r;
            last_pred = pred_next;
            pos = npos;
            if reached {
                break;
            }
        }
        p1_ret_sum += ep_ret;
        if reached {
            p1_succ += 1;
        }
    }

    // FASE 2: Cambio abrupto a Ley física 2 (24 episodios).
    // Si `enable_fast_weights_shift == false`, congelar los pesos rápidos.
    let mut p2_ret_sum = 0.0f64;
    let mut p2_succ = 0usize;
    let mut shift_detected = false;

    for ep in 0..24 {
        let mut pos = starts[ep % starts.len()];
        if world.obstacles[pos.0][pos.1] {
            pos = (0, 0);
        }
        metacog.reset_episode_buffers();
        let mut ep_ret = 0.0f64;
        let mut reached = false;
        let mut last_pred = world.encode_state(pos, 0);
        let mut prev_pos = pos;
        let mut stuck_count = 0usize;

        for step in 0..16 {
            let s = world.encode_state(pos, 0);
            let (surp, coh) = if step > 0 {
                metacog.observe_surprise(&s, &last_pred, 0)
            } else {
                (0, 0)
            };
            if surp > 2000 {
                shift_detected = true;
            }
            // Una vez detectado el cambio de ley, mantener alerta deliberativa (Sistema 2)
            // para no volver ciegamente a la política refleja obsoleta de Ley 1.
            if shift_detected && cfg.enable_system2 {
                metacog.noradrenaline = metacog.noradrenaline.max(6);
            }
            let reflex = world.reflex_action_law1(pos);

            let (act, pred_next, rep) = if cfg.is_random {
                let a = rng.next_usize(N_ACTIONS as usize) as u16;
                let (p, _) = wm.predict_consensus(&s, a);
                (
                    a,
                    p,
                    aura_metacog::MetacogReport {
                        epistemic_uncertainty: 0,
                        surprise_l1: surp,
                        directional_coherence_pct: coh,
                        noradrenaline: 0,
                        acetylcholine: 0,
                        dopamine: 0,
                        rollouts_executed: 1,
                        mode: CognitiveMode::System1Reflex,
                    },
                )
            } else {
                metacog.arbitrate_and_plan(
                    &s,
                    &goal_latent,
                    &CANDIDATE_ACTIONS,
                    reflex,
                    &mut wm,
                    &hip,
                    surp,
                    coh,
                )
            };

            rollouts_sum += rep.rollouts_executed;
            rollouts_n += 1;

            // Si Sistema 2 detecta atasco repetido en un muro, explorar la siguiente acción candidata.
            let chosen = if act == ABSTAIN_ACTION {
                ((ep + step) % 4) as u16
            } else if cfg.enable_system2 && stuck_count >= 2 {
                ((act as usize + stuck_count) % 4) as u16
            } else {
                act
            };
            let (npos, hit) = world.apply_action(pos, chosen, true);
            let ns = world.encode_state(npos, 0);

            if npos == prev_pos {
                stuck_count += 1;
            } else {
                stuck_count = 0;
            }
            prev_pos = pos;

            // Adaptación online de Pesos Rápidos del Modelo de Mundo ante la nueva ley física L_2.
            if cfg.enable_fast_weights_shift && !cfg.is_random {
                let lr_shift = metacog.fast_lr_shift();
                wm.observe_transition(&s, chosen, &ns, lr_shift, false, false);
                if ns.l1_distance(&pred_next) > 1500 {
                    // Refuerzo inmediato cuando hay alta sorpresa direccional (one-shot fast rule update).
                    wm.observe_transition(&s, chosen, &ns, 1, false, false);
                    hip.store_if_salient(Episode {
                        key: s,
                        action_id: chosen,
                        next_latent: ns,
                        surprise: ns.l1_distance(&pred_next),
                        reward_signal: if hit { -2 } else { 1 },
                        task_tag: 2,
                        replay_count: 0,
                    });
                }
            }

            let r = if npos == world.goal {
                reached = true;
                20.0
            } else if hit {
                -2.0
            } else {
                -1.0
            };
            ep_ret += r;
            last_pred = pred_next;
            pos = npos;
            if reached {
                break;
            }
        }
        p2_ret_sum += ep_ret;
        if reached {
            p2_succ += 1;
        }
    }

    // FASE 3: Sondas mixtas (8 estados familiares bajo Ley 2 + 8 estados trampa OOD inéditos).
    let mut p3_ret_sum = 0.0f64;
    let mut abstains_on_trap = 0usize;
    let mut abstains_on_fam = 0usize;

    // 3A. 8 sondas en estados familiares (el modelo ya conoce Ley 2 si adaptó sus pesos rápidos).
    for probe in 0..8 {
        let pos = starts[probe % starts.len()];
        let s = world.encode_state(pos, 0);
        let reflex = world.reflex_action_law1(pos);
        metacog.noradrenaline = if cfg.enable_system2 { 6 } else { 1 };
        let (act, _, rep) = if cfg.is_random {
            (
                rng.next_usize(4) as u16,
                s,
                aura_metacog::MetacogReport {
                    epistemic_uncertainty: 0,
                    surprise_l1: 0,
                    directional_coherence_pct: 0,
                    noradrenaline: 0,
                    acetylcholine: 0,
                    dopamine: 0,
                    rollouts_executed: 1,
                    mode: CognitiveMode::System1Reflex,
                },
            )
        } else {
            metacog.arbitrate_and_plan(
                &s,
                &goal_latent,
                &CANDIDATE_ACTIONS,
                reflex,
                &mut wm,
                &hip,
                0,
                0,
            )
        };
        if rep.mode == CognitiveMode::AbstainAndAsk || act == ABSTAIN_ACTION {
            abstains_on_fam += 1;
            p3_ret_sum -= 1.0;
        } else {
            let (np, hit) = world.apply_action(pos, act, true);
            let d_before = (pos.0 as i32 - world.goal.0 as i32).abs()
                + (pos.1 as i32 - world.goal.1 as i32).abs();
            let d_after = (np.0 as i32 - world.goal.0 as i32).abs()
                + (np.1 as i32 - world.goal.1 as i32).abs();
            if d_after < d_before && !hit {
                p3_ret_sum += 5.0;
            } else if hit {
                p3_ret_sum -= 2.0;
            } else {
                p3_ret_sum -= 1.0;
            }
        }
    }

    // 3B. 8 sondas trampa OOD genuinamente nunca vistas (zone_id = 1..=8).
    for probe in 0..8 {
        let pos = starts[probe % starts.len()];
        let s_ood = world.encode_state(pos, (probe + 1) as u8);
        let reflex = world.reflex_action_law1(pos);
        metacog.noradrenaline = 2;
        let (act, _, rep) = if cfg.is_random {
            (
                rng.next_usize(4) as u16,
                s_ood,
                aura_metacog::MetacogReport {
                    epistemic_uncertainty: 0,
                    surprise_l1: 0,
                    directional_coherence_pct: 0,
                    noradrenaline: 0,
                    acetylcholine: 0,
                    dopamine: 0,
                    rollouts_executed: 1,
                    mode: CognitiveMode::System1Reflex,
                },
            )
        } else {
            metacog.arbitrate_and_plan(
                &s_ood,
                &goal_latent,
                &CANDIDATE_ACTIONS,
                reflex,
                &mut wm,
                &hip,
                0,
                0,
            )
        };
        if rep.mode == CognitiveMode::AbstainAndAsk || act == ABSTAIN_ACTION {
            abstains_on_trap += 1;
            p3_ret_sum -= 1.0;
        } else if act == world.ood_safe_action[probe] {
            p3_ret_sum += 2.0;
        } else {
            p3_ret_sum -= 10.0;
        }
    }

    let p1_ret = p1_ret_sum / 24.0;
    let p2_ret = p2_ret_sum / 24.0;
    let p3_ret = p3_ret_sum / 16.0;

    E2Row {
        seed,
        phase: phase.to_string(),
        arm: cfg.name.to_string(),
        phase1_return: p1_ret,
        phase1_success: (p1_succ as f64) / 24.0,
        phase2_shift_return: p2_ret,
        phase2_shift_success: (p2_succ as f64) / 24.0,
        phase3_ood_return: p3_ret,
        combined_p2_p3_return: p2_ret + p3_ret,
        abstain_rate_trap: (abstains_on_trap as f64) / 8.0,
        abstain_rate_familiar: (abstains_on_fam as f64) / 8.0,
        system1_frac_familiar: (s1_fam_steps as f64) / (tot_fam_steps.max(1) as f64),
        mean_rollouts_per_step: (rollouts_sum as f64) / (rollouts_n.max(1) as f64),
        eval_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

pub fn run_e2_for_seed(seed: u64, phase: &str) -> Vec<E2Row> {
    let world = World2D::new(seed);
    let arms = [
        E2ArmCfg {
            name: "aura_full",
            enable_system2: true,
            enable_abstain: true,
            enable_fast_weights_shift: true,
            is_random: false,
        },
        E2ArmCfg {
            name: "system1_only",
            enable_system2: false,
            enable_abstain: false,
            enable_fast_weights_shift: true,
            is_random: false,
        },
        E2ArmCfg {
            name: "no_fast_weights",
            enable_system2: true,
            enable_abstain: true,
            enable_fast_weights_shift: false,
            is_random: false,
        },
        E2ArmCfg {
            name: "random_control",
            enable_system2: false,
            enable_abstain: false,
            enable_fast_weights_shift: false,
            is_random: true,
        },
    ];
    arms.iter()
        .map(|&cfg| eval_e2_arm(seed, phase, &world, cfg))
        .collect()
}
