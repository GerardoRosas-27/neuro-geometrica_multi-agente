//! Protocolo Canónico de Evaluación y Generador de Datos Constelación v4
//!
//! Genera constelaciones geométricas sintéticas de 6 puntos (12 coordenadas)
//! sometidas a transformaciones afines, rotaciones ortogonales y dinámicas no lineales:
//! - Familias conocidas: {0, 1, 2, 3, 4, 5, 6, 7}
//! - Familias nuevas fuera de distribución (OOD): {8, 10, 11} (Duffing, Lorenz-like, Van der Pol)
//! - Tarea E33: Predicción a horizonte h in {1, 2, 4, 8, 16} bajo contexto frío
//! - Métricas cuantitativas: Precisión en h=16 (rel_err < 0.05), log error relativo, cumplimiento Dale.

use crate::rng::Xoshiro256StarStar;
use std::f64::consts::PI;

pub const K_POINTS: usize = 6;
pub const K_OBS: usize = 4;
pub const HORIZON: usize = 16;
pub const REL_ERR_OK: f64 = 0.05;
pub const NEW_FAMILIES: [usize; 3] = [8, 10, 11];

/// Episodio de trayectoria geométrica de longitud total T
#[derive(Debug, Clone)]
pub struct TrajectoryEpisode {
    pub family_id: usize,
    pub is_new_family: bool,
    pub trajectory: Vec<Vec<f64>>,
    pub context: Vec<f64>,
}

/// Generador de episodios geométricos
#[derive(Debug, Clone)]
pub struct Cst4DataGenerator {
    pub n_points: usize,
}

impl Cst4DataGenerator {
    pub fn new() -> Self {
        Self { n_points: K_POINTS }
    }

    pub fn generate_episode(
        &self,
        family_id: usize,
        total_steps: usize,
        rng: &mut Xoshiro256StarStar,
    ) -> TrajectoryEpisode {
        let shape = family_id % 3;
        let (frames, ctx) = generate_canonical_trajectory(family_id, shape, total_steps, rng);
        let is_new = family_id == 8 || family_id == 10 || family_id == 11;
        TrajectoryEpisode {
            family_id,
            is_new_family: is_new,
            trajectory: frames,
            context: ctx,
        }
    }
}

/// Muestrea un parámetro en [0.02, 0.85] excluyendo la zona intermedia [0.10, 0.45]
pub fn sample_with_hole(rng: &mut Xoshiro256StarStar) -> f64 {
    loop {
        let val = 0.02 + (0.85 - 0.02) * rng.next_f64();
        if !(0.10..=0.45).contains(&val) {
            return val;
        }
    }
}

/// Genera un objeto geométrico base de 6 puntos (12 coordenadas)
pub fn generate_object(shape_type: usize, rng: &mut Xoshiro256StarStar) -> Vec<f64> {
    let mut pts = vec![0.0; K_POINTS * 2];
    let s = shape_type % 3;

    if s == 0 {
        let r = 0.6 + 0.3 * rng.next_f64();
        let phase = 2.0 * PI * rng.next_f64();
        for i in 0..K_POINTS {
            let theta = phase + (i as f64 * 2.0 * PI) / (K_POINTS as f64);
            pts[i * 2] = r * theta.cos();
            pts[i * 2 + 1] = r * theta.sin();
        }
    } else if s == 1 {
        let angles = [0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0];
        let v: Vec<(f64, f64)> = angles.iter().map(|&a| (0.7 * a.cos(), 0.7 * a.sin())).collect();
        pts[0] = v[0].0; pts[1] = v[0].1;
        pts[2] = v[1].0; pts[3] = v[1].1;
        pts[4] = v[2].0; pts[5] = v[2].1;
        pts[6] = 0.5 * (v[0].0 + v[1].0); pts[7] = 0.5 * (v[0].1 + v[1].1);
        pts[8] = 0.5 * (v[1].0 + v[2].0); pts[9] = 0.5 * (v[1].1 + v[2].1);
        pts[10] = 0.5 * (v[2].0 + v[0].0); pts[11] = 0.5 * (v[2].1 + v[0].1);
    } else {
        for i in 0..K_POINTS {
            let r = 0.4 + 0.4 * rng.next_f64();
            let theta = (i as f64 * 2.0 * PI) / (K_POINTS as f64) + (rng.next_f64() - 0.5) * 0.2;
            pts[i * 2] = r * theta.cos();
            pts[i * 2 + 1] = r * theta.sin();
        }
    }

    pts
}

/// Aplica la dinámica geométrica o caótica de la familia seleccionada
pub fn apply_dynamics(family: usize, x: f64, y: f64, param: f64, dt: f64) -> (f64, f64) {
    match family {
        0 => (x + 0.01 * dt, y - 0.01 * dt),
        1 => {
            let omega = param * dt;
            (x * omega.cos() - y * omega.sin(), x * omega.sin() + y * omega.cos())
        }
        2 => (-x, y),
        3 => {
            let s = 1.0 + (param - 0.4) * 0.2 * dt;
            (x * s, y * s)
        }
        4 => (x + param * dt * y, y),
        5 => {
            let phi = (param * 0.3 * dt).clamp(-0.4, 0.4);
            (x * phi.cosh() + y * phi.sinh(), x * phi.sinh() + y * phi.cosh())
        }
        6 => {
            let omega = param * dt;
            let s = 1.0 - 0.05 * dt;
            (s * (x * omega.cos() - y * omega.sin()), s * (x * omega.sin() + y * omega.cos()))
        }
        7 => {
            let r2 = x * x + y * y;
            let dr = 0.5 * (1.0 - r2) * dt;
            let omega = (param + 0.5) * dt;
            (
                (x + dr * x) * omega.cos() - (y + dr * y) * omega.sin(),
                (x + dr * x) * omega.sin() + (y + dr * y) * omega.cos(),
            )
        }
        8 => {
            // Duffing oscillator (OOD)
            let dx = y;
            let dy = x - x * x * x - 0.2 * y;
            (x + dx * dt * param, y + dy * dt * param)
        }
        10 => {
            // Lorenz-like coupled dynamics (OOD)
            let dx = 10.0 * (y - x);
            let dy = x * 27.0 - y;
            (x + dx * 0.005 * param, y + dy * 0.005 * param)
        }
        _ => {
            // 11: Van der Pol oscillator (OOD)
            let mu = param * 1.5;
            let dx = y;
            let dy = mu * (1.0 - x * x) * y - x;
            (x + dx * 0.04, y + dy * 0.04)
        }
    }
}

/// Generador canónico de trayectorias
pub fn generate_canonical_trajectory(
    family: usize,
    shape_type: usize,
    total_steps: usize,
    rng: &mut Xoshiro256StarStar,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut frames = Vec::with_capacity(total_steps);
    let mut cur = generate_object(shape_type, rng);
    let param = sample_with_hole(rng);
    frames.push(cur.clone());

    for _ in 1..total_steps {
        let mut nxt = vec![0.0; K_POINTS * 2];
        for p in 0..K_POINTS {
            let (nx, ny) = apply_dynamics(family, cur[p * 2], cur[p * 2 + 1], param, 0.1);
            nxt[p * 2] = nx;
            nxt[p * 2 + 1] = ny;
        }
        frames.push(nxt.clone());
        cur = nxt;
    }

    let mut ctx = vec![0.0; 10];
    let f0 = &frames[0];
    let f1 = &frames[1];
    for i in 0..5 {
        ctx[i * 2] = (f1[i * 2] - f0[i * 2]).clamp(-1.0, 1.0);
        ctx[i * 2 + 1] = (f1[i * 2 + 1] - f0[i * 2 + 1]).clamp(-1.0, 1.0);
    }

    (frames, ctx)
}

/// Calcula el error relativo euclidiano L2 normalizado
pub fn compute_relative_error(pred: &[f64], target: &[f64]) -> f64 {
    let mut num_sq = 0.0;
    let mut den_sq = 0.0;
    for i in 0..pred.len() {
        let diff = pred[i] - target[i];
        num_sq += diff * diff;
        den_sq += target[i] * target[i];
    }
    num_sq.sqrt() / (den_sq.sqrt() + 1e-6)
}

/// Métrica de Error Relativo Logarítmico entre trayectoria predicha y real
pub fn compute_log_relative_error(pred: &[f64], target: &[f64]) -> f64 {
    let rel_err = compute_relative_error(pred, target);
    (rel_err + 1e-12).ln()
}
