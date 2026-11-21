//! Experimento E3 (Etapa 2): Validación del Ciclo de Sueño (Anti-Olvido en Planos de Bits Ternarios).
//!
//! Protocolo exacto de §5 Etapa 2:
//! 1. Enseña Tarea A (64 train / 128 test disjunto) en `TernaryMatrix` (`W_slow`, 2 bits/peso)
//!    y guarda un presupuesto acotado de episodios salientes en `HippocampusGraph`.
//! 2. Expone ÚNICAMENTE Tarea B (64 train / 128 test disjunto, con subespacio de entrada solapado
//!    para generar interferencia real) sin acceso directo al dataset de A.
//! 3. Evalúa `W_slow` puro (`project_zero_mul`, con `FastWeightMemory` en cero) sobre `Test_A` y `Test_B`.
//!
//! Brazos pareados (todos parten del MISMO checkpoint post-Tarea A):
//! `aura_sleep_full`, `ablation_no_replay`, `ablation_no_hysteresis`, `ablation_naive`.

use aura_hippocampus::{Episode, HippocampusGraph, SleepConfig, SleepConsolidator};
use aura_math::{LatentVec, TernaryMatrix, Xorshift64, LATENT_DIM};

const OUT_DIM: usize = 16;

#[derive(Clone, Debug)]
pub struct E3Row {
    pub seed: u64,
    pub phase: String,
    pub arm: String,
    pub acc_a_pre: f64,
    pub acc_a_post: f64,
    pub forgetting_a: f64,
    pub acc_b_post: f64,
    pub combined_ab: f64,
    pub weight_flips: usize,
    pub nonzero_frac: f64,
    pub eval_ms: f64,
}

struct TaskData {
    train: Vec<Episode>,
    test: Vec<(LatentVec, LatentVec)>,
}

fn generate_task(
    rng: &mut Xorshift64,
    active_start: usize,
    active_end: usize,
    task_tag: u8,
    n_train: usize,
    n_test: usize,
) -> TaskData {
    // Regla generadora oculta ternaria dispersa (16 salidas x 256 entradas):
    // exactamente 12 características informativas por salida dentro del subespacio activo
    // (128 dimensiones se solapan entre Tarea A y Tarea B -> interferencia genuina).
    let span = active_end - active_start;
    let mut gt = TernaryMatrix::new_zeros(OUT_DIM, LATENT_DIM);
    for r in 0..OUT_DIM {
        let mut placed = 0;
        while placed < 12 {
            let c = active_start + rng.next_usize(span);
            if gt.weight_at(r, c) == 0 {
                let sign = if (rng.next_u64() & 1) == 0 { 1i8 } else { -1i8 };
                gt.set_weight_and_shadow(r, c, sign, (sign as i16) * 128);
                placed += 1;
            }
        }
    }

    let make_sample = |r_gen: &mut Xorshift64| -> (LatentVec, LatentVec) {
        let mut x = LatentVec::zero();
        for c in active_start..active_end {
            let d = r_gen.next_u64() & 3;
            x.data[c] = match d {
                0 => 48,
                1 => -48,
                2 => 32,
                _ => -32,
            };
        }
        let mut raw_y = [0i32; OUT_DIM];
        gt.project_zero_mul(&x.data, &mut raw_y);
        let mut y = LatentVec::zero();
        for r in 0..OUT_DIM {
            y.data[r] = if raw_y[r] >= 0 { 64 } else { -64 };
        }
        (x, y)
    };

    let mut train = Vec::with_capacity(n_train);
    for i in 0..n_train {
        let (k, y) = make_sample(rng);
        train.push(Episode {
            key: k,
            action_id: task_tag as u16,
            next_latent: y,
            surprise: 1000 + (i as i32),
            reward_signal: 1,
            task_tag,
            replay_count: 0,
        });
    }

    let mut test = Vec::with_capacity(n_test);
    for _ in 0..n_test {
        test.push(make_sample(rng));
    }
    TaskData { train, test }
}

/// Precisión de signo de `TernaryMatrix::project_zero_mul` puro sobre un conjunto held-out.
fn eval_ternary_sign_accuracy(mat: &TernaryMatrix, test: &[(LatentVec, LatentVec)]) -> f64 {
    let mut pred = [0i32; OUT_DIM];
    let mut correct = 0usize;
    let mut total = 0usize;
    for (x, y) in test {
        mat.project_zero_mul(&x.data, &mut pred);
        for r in 0..OUT_DIM {
            let p_pos = pred[r] >= 0;
            let y_pos = y.data[r] >= 0;
            if p_pos == y_pos {
                correct += 1;
            }
            total += 1;
        }
    }
    (correct as f64) / (total.max(1) as f64)
}

pub fn run_e3_for_seed(seed: u64, phase: &str) -> Vec<E3Row> {
    let mut rng_data = Xorshift64::from_seed(seed).split_stream(0xE300_DA7A);
    // Tarea A activa en dims [0..96); Tarea B activa en dims [32..128).
    // 64 de las 96 dimensiones activas (66.7%) se SOLAPAN -> interferencia severa si se sobrescribe,
    // mientras n_train=256 > d=96 permite generalización alta (~85%+) sobre Test_A y Test_B disjuntos.
    let task_a = generate_task(&mut rng_data, 0, 96, 1, 256, 128);
    let task_b = generate_task(&mut rng_data, 32, 128, 2, 256, 128);

    // Paso 1: Consolidar Tarea A en `W_slow` base (compartido por los 4 brazos para contraste pareado).
    let mut base_mat = TernaryMatrix::new_zeros(OUT_DIM, LATENT_DIM);
    let mut empty_hip = HippocampusGraph::new(128, 0, 0);
    let mut rng_a = Xorshift64::from_seed(seed).split_stream(0xE300_A001);
    let cfg_a = SleepConfig {
        cycles: 2400,
        use_replay: false,
        replay_ratio_pct: 0,
        hysteresis_hi: 96,
        hysteresis_lo: 24,
        shadow_cap: 512,
        step_delta: 12,
        target_scale_shift: 3,
        key_active_thresh: 16,
        err_deadband: 16,
    };
    SleepConsolidator::consolidate_into_ternary(
        &mut base_mat,
        &mut empty_hip,
        &task_a.train,
        cfg_a,
        &mut rng_a,
    );
    let acc_a_pre = eval_ternary_sign_accuracy(&base_mat, &task_a.test);

    // Guardar presupuesto comprimido (96 de 256 episodios de Tarea A = 37.5%) en el hipocampo.
    let mut base_hip = HippocampusGraph::new(128, 0, 0);
    for ep in task_a.train.iter().take(96) {
        base_hip.insert_episode(ep.clone());
    }

    // Paso 2: Evaluar los 4 brazos pareados sobre Tarea B (cero acceso al dataset de Tarea A).
    struct ArmSpec {
        name: &'static str,
        use_replay: bool,
        use_hysteresis: bool,
    }
    let specs = [
        ArmSpec {
            name: "aura_sleep_full",
            use_replay: true,
            use_hysteresis: true,
        },
        ArmSpec {
            name: "ablation_no_replay",
            use_replay: false,
            use_hysteresis: true,
        },
        ArmSpec {
            name: "ablation_no_hysteresis",
            use_replay: true,
            use_hysteresis: false,
        },
        ArmSpec {
            name: "ablation_naive",
            use_replay: false,
            use_hysteresis: false,
        },
    ];

    let mut rows = Vec::with_capacity(specs.len());
    for spec in specs {
        let t0 = std::time::Instant::now();
        let mut mat = base_mat.clone();
        let mut hip = base_hip.clone();
        if !spec.use_hysteresis {
            // Al desactivar la histéresis, los acumuladores latentes pierden el margen de inercia
            // (se fijan al signo inmediato {-1, 0, +1}) para aislar el efecto puro de la histéresis.
            for r in 0..OUT_DIM {
                for c in 0..LATENT_DIM {
                    let w = mat.weight_at(r, c);
                    mat.set_weight_and_shadow(r, c, w, w as i16);
                }
            }
        }

        let sleep_cfg = SleepConfig {
            cycles: 2400,
            use_replay: spec.use_replay,
            replay_ratio_pct: if spec.use_replay { 50 } else { 0 },
            hysteresis_hi: if spec.use_hysteresis { 96 } else { 1 },
            hysteresis_lo: if spec.use_hysteresis { 24 } else { 0 },
            shadow_cap: if spec.use_hysteresis { 512 } else { 2 },
            step_delta: if spec.use_hysteresis { 12 } else { 2 },
            target_scale_shift: 3,
            key_active_thresh: 16,
            err_deadband: 16,
        };
        let mut rng_b = Xorshift64::from_seed(seed).split_stream(0xE300_B002);
        let rep = SleepConsolidator::consolidate_into_ternary(
            &mut mat,
            &mut hip,
            &task_b.train,
            sleep_cfg,
            &mut rng_b,
        );

        // Paso 3: Evaluar `W_slow` puro (0-mul, FastWeights = 0) en Test_A y Test_B no vistos.
        let acc_a_post = eval_ternary_sign_accuracy(&mat, &task_a.test);
        let acc_b_post = eval_ternary_sign_accuracy(&mat, &task_b.test);
        let nnz_frac = (mat.nonzero_count() as f64) / ((OUT_DIM * LATENT_DIM) as f64);

        rows.push(E3Row {
            seed,
            phase: phase.to_string(),
            arm: spec.name.to_string(),
            acc_a_pre,
            acc_a_post,
            forgetting_a: acc_a_pre - acc_a_post,
            acc_b_post,
            combined_ab: 0.5 * (acc_a_post + acc_b_post),
            weight_flips: rep.weight_flips,
            nonzero_frac: nnz_frac,
            eval_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }
    rows
}
