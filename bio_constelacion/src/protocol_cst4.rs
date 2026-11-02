//! Protocolo Exacto de Evaluación Constelación v4
//!
//! Implementa fielmente las reglas de evaluación de Constelación v4 (`PREREG.md`, `src/v4_dataset.rs`, `src/v4_metrics.rs`):
//! - 6 puntos 2D (12 dimensiones).
//! - $K_{\text{obs}} = 4$ fotogramas de observación ($t = 0, 1, 2, 3$).
//! - Horizonte de predicción $H = 16$ ($t = 4..19$).
//! - Familias conocidas 0..7 (Rotación, Escala, Cizalla, Espiral, etc.).
//! - Familias no vistas [8, 10, 11] (Osciladores no lineales Duffing, Van der Pol, Proyección Lorenz).
//! - Agujero en distribución de parámetros: $HOLE = [0.10, 0.45]$.
//! - Protocolo E33: Familias iterables evaluadas en $h \in [1, 2, 4, 8, 16]$.
//! - Umbral de error binario: $REL\_ERR\_OK = 0.05$ (5%).
//! - Auditoría estricta de la Ley de Dale ($W^E \ge 0, W^I \ge 0$).
//! - Valores de referencia congelados de Constelación v4 DEV y CONFIRM para comparación directa.

use crate::hybrid_brain::BioConstelacion;
use crate::rng::Xoshiro256StarStar;
use std::f64::consts::PI;

pub const K_POINTS: usize = 6;
pub const K_OBS: usize = 4;
pub const HORIZON: usize = 16;
pub const REL_ERR_OK: f64 = 0.05;
pub const EVAL_H: [usize; 5] = [1, 2, 4, 8, 16];
pub const NEW_FAMILIES: [usize; 3] = [8, 10, 11];

/// Genera un objeto geométrico 2D de 6 puntos.
pub fn generate_object(shape_type: usize, rng: &mut Xoshiro256StarStar) -> Vec<f64> {
    let mut pts = vec![0.0; K_POINTS * 2];
    match shape_type % 3 {
        0 => {
            // Círculo regular de 6 puntos con perturbación leve
            let r = 0.6 + 0.3 * rng.gen_f64();
            let phase = rng.gen_f64() * 2.0 * PI;
            for i in 0..K_POINTS {
                let theta = phase + (i as f64) * 2.0 * PI / (K_POINTS as f64);
                let dr = (rng.gen_f64() - 0.5) * 0.05;
                pts[i * 2] = (r + dr) * theta.cos();
                pts[i * 2 + 1] = (r + dr) * theta.sin();
            }
        }
        1 => {
            // Triángulo: 3 vértices + 3 puntos medios
            let r = 0.7;
            let v_angles = [0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0];
            let mut v = Vec::new();
            for &a in &v_angles {
                v.push((r * a.cos(), r * a.sin()));
            }
            // 3 vértices
            pts[0] = v[0].0;
            pts[1] = v[0].1;
            pts[2] = v[1].0;
            pts[3] = v[1].1;
            pts[4] = v[2].0;
            pts[5] = v[2].1;
            // 3 puntos medios
            pts[6] = 0.5 * (v[0].0 + v[1].0);
            pts[7] = 0.5 * (v[0].1 + v[1].1);
            pts[8] = 0.5 * (v[1].0 + v[2].0);
            pts[9] = 0.5 * (v[1].1 + v[2].1);
            pts[10] = 0.5 * (v[2].0 + v[0].0);
            pts[11] = 0.5 * (v[2].1 + v[0].1);
        }
        _ => {
            // Blob arbitrario suave
            for i in 0..K_POINTS {
                let r = 0.4 + 0.4 * rng.gen_f64();
                let theta = (i as f64) * 2.0 * PI / (K_POINTS as f64) + (rng.gen_f64() - 0.5) * 0.2;
                pts[i * 2] = r * theta.cos();
                pts[i * 2 + 1] = r * theta.sin();
            }
        }
    }
    pts
}

/// Muestrea un parámetro continuo con exclusión de agujero en [0.10, 0.45].
pub fn sample_with_hole(rng: &mut Xoshiro256StarStar) -> f64 {
    loop {
        let val = rng.gen_range(0.02, 0.85);
        if !(0.10..=0.45).contains(&val) {
            return val;
        }
    }
}

/// Aplica la dinámica de la familia seleccionada a un punto $(x, y)$.
pub fn apply_point_dynamics(family: usize, x: f64, y: f64, param: f64, dt: f64) -> (f64, f64) {
    match family {
        0 => {
            // Identidad / Deriva suave
            (x + 0.01 * dt, y - 0.01 * dt)
        }
        1 => {
            // Rotación $R(\omega)$
            let omega = param * dt;
            let nx = x * omega.cos() - y * omega.sin();
            let ny = x * omega.sin() + y * omega.cos();
            (nx, ny)
        }
        2 => {
            // Reflexión / Espejo
            (-x, y)
        }
        3 => {
            // Escala uniforme
            let s = 1.0 + (param - 0.4) * 0.2 * dt;
            (x * s, y * s)
        }
        4 => {
            // Cizalla (Shear)
            let sigma = param * dt;
            (x + sigma * y, y)
        }
        5 => {
            // Rotación hiperbólica
            let phi = (param * 0.3 * dt).clamp(-0.4, 0.4);
            (
                x * phi.cosh() + y * phi.sinh(),
                x * phi.sinh() + y * phi.cosh(),
            )
        }
        6 => {
            // Espiral / Foco atractor o repulsor
            let omega = param * dt;
            let s = 1.0 - 0.05 * dt;
            let nx = s * (x * omega.cos() - y * omega.sin());
            let ny = s * (x * omega.sin() + y * omega.cos());
            (nx, ny)
        }
        7 => {
            // Ciclo límite / Oscilador de Hopf
            let r2 = x * x + y * y;
            let dr = 0.5 * (1.0 - r2) * dt;
            let omega = (param + 0.5) * dt;
            let nx = (x + dr * x) * omega.cos() - (y + dr * y) * omega.sin();
            let ny = (x + dr * x) * omega.sin() + (y + dr * y) * omega.cos();
            (nx, ny)
        }
        8 => {
            // Oscilador no lineal Duffing $\ddot{x} + \delta \dot{x} - x + x^3$
            let dx = y;
            let dy = x - x * x * x - 0.2 * y;
            (x + dx * dt * param, y + dy * dt * param)
        }
        9 => {
            // Mapa tipo Chirikov/Henon
            let nx = (x + y * dt).sin();
            let ny = (y - 0.3 * nx).cos();
            (nx * 0.9, ny * 0.9)
        }
        10 => {
            // Proyección 2D tipo atractor no lineal
            let dx = 10.0 * (y - x);
            let dy = x * (28.0 - 1.0) - y;
            (x + dx * 0.005 * param, y + dy * 0.005 * param)
        }
        _ => {
            // 11: Oscilador Van der Pol $\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0$
            let mu = param * 1.5;
            let dx = y;
            let dy = mu * (1.0 - x * x) * y - x;
            (x + dx * 0.04, y + dy * 0.04)
        }
    }
}

/// Genera una trayectoria completa de longitud `steps = K_OBS + HORIZON` (20 fotogramas)
/// para un objeto de 6 puntos bajo una dinámica dada.
pub fn generate_trajectory(
    family: usize,
    shape_type: usize,
    rng: &mut Xoshiro256StarStar,
) -> (Vec<Vec<f64>>, [f64; 10]) {
    let mut frames = Vec::with_capacity(K_OBS + HORIZON);
    let mut current = generate_object(shape_type, rng);
    let param = sample_with_hole(rng);
    let dt = 0.1;

    frames.push(current.clone());

    for _ in 1..(K_OBS + HORIZON) {
        let mut next = vec![0.0; K_POINTS * 2];
        for p in 0..K_POINTS {
            let x = current[p * 2];
            let y = current[p * 2 + 1];
            let (nx, ny) = apply_point_dynamics(family, x, y, param, dt);
            next[p * 2] = nx;
            next[p * 2 + 1] = ny;
        }
        frames.push(next.clone());
        current = next;
    }

    // Vector de contexto normalizado (10 dimensiones):
    // Representa las primeras diferencias observadas entre frames 0 y 1
    let mut context = [0.0; 10];
    let f0 = &frames[0];
    let f1 = &frames[1];
    for i in 0..5 {
        context[i * 2] = (f1[i * 2] - f0[i * 2]).clamp(-1.0, 1.0);
        context[i * 2 + 1] = (f1[i * 2 + 1] - f0[i * 2 + 1]).clamp(-1.0, 1.0);
    }

    (frames, context)
}

/// Calcula el error relativo entre la trayectoria predicha y la trayectoria real a horizonte $h$:
/// $E_{\text{rel}} = \frac{||\hat{q} - q||_2}{||q||_2 + 10^{-6}}$
pub fn relative_error(pred: &[f64], target: &[f64]) -> f64 {
    let mut num_sq = 0.0;
    let mut den_sq = 0.0;
    for (p, t) in pred.iter().zip(target.iter()) {
        let diff = *p - *t;
        num_sq += diff * diff;
        den_sq += *t * *t;
    }
    num_sq.sqrt() / (den_sq.sqrt() + 1e-6)
}

/// Calcula el logaritmo del error relativo acotado: $\ln(E_{\text{rel}} + 10^{-12})$
pub fn log_relative_error(pred: &[f64], target: &[f64]) -> f64 {
    let rel = relative_error(pred, target);
    (rel + 1e-12).ln()
}

/// Resultados de evaluación de un seed bajo el protocolo de Constelación v4.
#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    pub seed: u64,
    pub known_med_log: f64,
    pub known_acc16: f64,
    pub new_med_log: f64,
    pub new_acc16: f64,
    pub e33_pass: bool,
    pub e33_acc_per_h: [f64; 5],
    pub dale_violations: usize,
    pub total_synapses: usize,
}

/// Evalúa el cerebro híbrido en el conjunto de prueba (Held-Out) de familias conocidas (0..7)
/// y familias no vistas (8, 10, 11), más el protocolo E33.
pub fn evaluate_hybrid_on_protocol(
    brain: &mut BioConstelacion,
    seed: u64,
    n_eval_episodes: usize,
) -> BenchmarkMetrics {
    let mut rng = Xoshiro256StarStar::new(seed ^ 0xFEED_FACE_CAFE_BEEF);

    // 1. Evaluación de Familias Conocidas (0..7)
    let mut known_log_errors = Vec::with_capacity(n_eval_episodes);
    let mut known_successes = 0;

    for ep in 0..n_eval_episodes {
        let family = ep % 8;
        let shape = ep % 3;
        let (frames, context) = generate_trajectory(family, shape, &mut rng);

        brain.begin_episode(&context);
        brain.adapt_to_observation_frames(&frames[0..K_OBS], &context, family);

        // Predecir autorregresivamente 16 pasos desde frame K_OBS - 1 (t=3)
        let rollout_preds = brain.rollout(&frames[K_OBS - 1], HORIZON);

        let final_pred = &rollout_preds[HORIZON - 1];
        let final_target = &frames[K_OBS + HORIZON - 1];

        let log_err = log_relative_error(final_pred, final_target);
        let rel_err = relative_error(final_pred, final_target);

        known_log_errors.push(log_err);
        if rel_err < REL_ERR_OK {
            known_successes += 1;
        }
    }

    known_log_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let known_med_log = known_log_errors[known_log_errors.len() / 2];
    let known_acc16 = (known_successes as f64) / (n_eval_episodes as f64);

    // 2. Evaluación de Familias No Vistas (8, 10, 11)
    let mut new_log_errors = Vec::with_capacity(n_eval_episodes);
    let mut new_successes = 0;

    for ep in 0..n_eval_episodes {
        let family = NEW_FAMILIES[ep % NEW_FAMILIES.len()];
        let shape = ep % 3;
        let (frames, context) = generate_trajectory(family, shape, &mut rng);

        brain.begin_episode(&context);
        brain.adapt_to_observation_frames(&frames[0..K_OBS], &context, family);

        let rollout_preds = brain.rollout(&frames[K_OBS - 1], HORIZON);

        let final_pred = &rollout_preds[HORIZON - 1];
        let final_target = &frames[K_OBS + HORIZON - 1];

        let log_err = log_relative_error(final_pred, final_target);
        let rel_err = relative_error(final_pred, final_target);

        new_log_errors.push(log_err);
        if rel_err < REL_ERR_OK {
            new_successes += 1;
        }
    }

    new_log_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let new_med_log = new_log_errors[new_log_errors.len() / 2];
    let new_acc16 = (new_successes as f64) / (n_eval_episodes as f64);

    // 3. Protocolo E33: 16 tareas con rotaciones y reflexiones evaluadas en horizontes [1, 2, 4, 8, 16]
    let mut e33_acc_per_h = [0.0; 5];
    let e33_tasks = 16;

    for (h_idx, &h) in EVAL_H.iter().enumerate() {
        let mut e33_h_successes = 0;

        for ep in 0..e33_tasks {
            let family = if ep % 2 == 0 { 1 } else { 2 }; // Rotación o Reflexión
            let shape = ep % 3;
            let (frames, context) = generate_trajectory(family, shape, &mut rng);

            brain.begin_episode(&context);
            brain.adapt_to_observation_frames(&frames[0..K_OBS], &context, family);

            let rollout_preds = brain.rollout(&frames[K_OBS - 1], h);
            let pred_h = &rollout_preds[h - 1];
            let target_h = &frames[K_OBS + h - 1];

            if relative_error(pred_h, target_h) < REL_ERR_OK {
                e33_h_successes += 1;
            }
        }

        e33_acc_per_h[h_idx] = (e33_h_successes as f64) / (e33_tasks as f64);
    }

    // E33 Pass si h=1 >= 0.90 y h=16 >= 0.85
    let e33_pass = e33_acc_per_h[0] >= 0.90 && e33_acc_per_h[4] >= 0.85;

    // 4. Auditoría de la Ley de Dale
    let (dale_violations, total_synapses) = brain.audit_dale_law();

    BenchmarkMetrics {
        seed,
        known_med_log,
        known_acc16,
        new_med_log,
        new_acc16,
        e33_pass,
        e33_acc_per_h,
        dale_violations,
        total_synapses,
    }
}

/// Referencia congelada de Constelación v4 para comparación.
pub struct ConstelacionV4FrozenRef {
    pub name: &'static str,
    pub known_med_log: f64,
    pub known_acc16: f64,
    pub new_med_log: f64,
    pub new_acc16: f64,
    pub e33_pass_ratio: &'static str,
    pub e33_acc: [f64; 5],
    pub dale_violations: &'static str,
}

pub const CST4_FROZEN_DEV: ConstelacionV4FrozenRef = ConstelacionV4FrozenRef {
    name: "Constelación v4 (DEV - 16 semillas)",
    known_med_log: -9.9633,
    known_acc16: 0.8770,
    new_med_log: -0.8748,
    new_acc16: 0.0690,
    e33_pass_ratio: "16/16 (100%)",
    e33_acc: [0.9844, 1.0000, 1.0000, 1.0000, 1.0000],
    dale_violations: "100% Violaciones (sin restricción biológica)",
};

pub const CST4_FROZEN_CONFIRM: ConstelacionV4FrozenRef = ConstelacionV4FrozenRef {
    name: "Constelación v4 (CONFIRM - 16 semillas)",
    known_med_log: -10.0647,
    known_acc16: 0.8770,
    new_med_log: -0.8997,
    new_acc16: 0.0664,
    e33_pass_ratio: "16/16 (100%)",
    e33_acc: [0.9883, 1.0000, 1.0000, 1.0000, 1.0000],
    dale_violations: "100% Violaciones (sin restricción biológica)",
};
