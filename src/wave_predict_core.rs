//! Core de predicción por **interferencia de ondas** (sin RQM).
//!
//! Idea:
//! - Ondas del **pasado** (observaciones) se inyectan en el líquido.
//! - Cada hipótesis del **futuro** es otra onda.
//! - Donde pasado y futuro **interfieren constructivamente**, la predicción
//!   se considera correcta; el argmax de esa puntuación es el resultado.
//!
//! Camino óptimo (latencia):
//! 1. **Score analítico** de paquetes gaussianos (cerrado, sin PDE) — default.
//! 2. Opcional: superposición en rejilla chica 2D + pocos pasos lineales.
//!
//! No usa `NativeThermoRqmEprSubstrate` ni el NLS 16³ completo.

use num_complex::Complex64;
use std::f64::consts::PI;
use std::time::Instant;

/// Paquete de onda gaussiano (envolvente compleja).
#[derive(Clone, Copy, Debug)]
pub struct WavePacket {
    pub id: u64,
    pub amp: f64,
    pub sigma: f64,
    pub center: [f64; 3],
    pub k: [f64; 3],
    /// Fase global (rad). Codifica canal / contenido.
    pub phase: f64,
    /// Marca temporal relativa (pasado < 0, futuro > 0), solo metadato.
    pub time_tag: f64,
}

impl WavePacket {
    pub fn past(id: u64, content: usize, amp: f64) -> Self {
        let (c, k, phase) = content_geometry(content, false);
        Self {
            id,
            amp,
            sigma: 1.35,
            center: c,
            k,
            phase,
            time_tag: -1.0,
        }
    }

    pub fn future(id: u64, content: usize, amp: f64) -> Self {
        let (c, k, phase) = content_geometry(content, true);
        Self {
            id,
            amp,
            sigma: 1.35,
            center: c,
            k,
            phase,
            time_tag: 1.0,
        }
    }
}

/// Geometría compartida: el futuro “correcto” del contenido `c` se alinea en
/// fase/centro con el pasado del mismo `c` (continuación coherente).
fn content_geometry(content: usize, is_future: bool) -> ([f64; 3], [f64; 3], f64) {
    let c = content % 8;
    let angle = c as f64 * (PI / 4.0);
    let radius = 2.2;
    // Misma fase/canal para pasado y futuro del mismo contenido → cos(Δφ)≈1 (construye).
    // El futuro se adelanta un poco a lo largo de k (propagación).
    let drift = if is_future { 0.85 } else { 0.0 };
    let cx = 8.0 + radius * angle.cos() + drift * angle.cos();
    let cy = 8.0 + radius * angle.sin() + drift * angle.sin();
    let cz = 8.0;
    let k = [0.15 * angle.cos(), 0.15 * angle.sin(), 0.0];
    let phase = angle; // canal = contenido
    ([cx, cy, cz], k, phase)
}

/// Score de interferencia constructiva entre dos paquetes (forma cerrada).
///
/// Aproxima ∫ Re(ψ_a* ψ_b) dV para gaussianas: amplitud × overlap espacial ×
/// cos(Δφ + k·Δx). >0 construye, <0 destruye.
pub fn analytic_interference(a: &WavePacket, b: &WavePacket) -> f64 {
    let mut dx2 = 0.0;
    let mut k_dot_dx = 0.0;
    for i in 0..3 {
        let d = b.center[i] - a.center[i];
        dx2 += d * d;
        k_dot_dx += 0.5 * (a.k[i] + b.k[i]) * d;
    }
    let sig2 = a.sigma * a.sigma + b.sigma * b.sigma;
    let spatial = (-dx2 / sig2).exp();
    let dphase = b.phase - a.phase + k_dot_dx;
    a.amp * b.amp * spatial * dphase.cos()
}

/// Energía de interferencia de un futuro contra todo el pasado.
pub fn score_future_against_past(past: &[WavePacket], future: &WavePacket) -> f64 {
    past.iter().map(|p| analytic_interference(p, future)).sum()
}

#[derive(Clone, Debug, Default)]
pub struct WavePredictCore {
    pub past: Vec<WavePacket>,
    pub next_id: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct PredictionReport {
    pub best_future_id: u64,
    pub best_content: usize,
    pub best_score: f64,
    pub scores: usize,
}

impl WavePredictCore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear_past(&mut self) {
        self.past.clear();
    }

    pub fn inject_past_content(&mut self, content: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.past.push(WavePacket::past(id, content, 1.0));
        id
    }

    pub fn inject_past_packet(&mut self, packet: WavePacket) {
        self.past.push(packet);
    }

    /// Máquina de predicción: entre hipótesis de futuro, gana la de mayor
    /// interferencia constructiva con el pasado acumulado.
    pub fn predict_best_future(&self, future_contents: &[usize]) -> PredictionReport {
        assert!(
            !future_contents.is_empty(),
            "need at least one future hypothesis"
        );
        let mut best_i = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_id = 0u64;
        for (i, &content) in future_contents.iter().enumerate() {
            let fut = WavePacket::future(i as u64, content, 1.0);
            let s = score_future_against_past(&self.past, &fut);
            if s > best_score {
                best_score = s;
                best_i = i;
                best_id = fut.id;
            }
        }
        PredictionReport {
            best_future_id: best_id,
            best_content: future_contents[best_i],
            best_score,
            scores: future_contents.len(),
        }
    }

    /// Atajo: pasado = observación `obs`; futuros = candidatos; devuelve el
    /// candidato que mejor interfiere (predicción “correcta” si está en la lista).
    pub fn predict_from_observation(
        &mut self,
        observation: usize,
        candidates: &[usize],
    ) -> PredictionReport {
        self.clear_past();
        self.inject_past_content(observation);
        self.predict_best_future(candidates)
    }
}

// ─── Refinamiento opcional en rejilla 2D chica (lineal, pocos pasos) ────────

pub const GRID: usize = 24;
pub const CELLS2: usize = GRID * GRID;

#[derive(Clone, Debug)]
pub struct WaveField2D {
    pub psi: Vec<Complex64>,
    pub dt: f64,
    pub alpha: f64,
}

impl WaveField2D {
    pub fn new() -> Self {
        Self {
            psi: vec![Complex64::new(0.0, 0.0); CELLS2],
            dt: 0.04,
            alpha: 0.4,
        }
    }

    pub fn clear(&mut self) {
        for z in &mut self.psi {
            *z = Complex64::new(0.0, 0.0);
        }
    }

    fn ix(i: usize, j: usize) -> usize {
        i + GRID * j
    }

    pub fn add_packet(&mut self, p: &WavePacket) {
        let inv = 1.0 / (2.0 * p.sigma * p.sigma);
        for j in 0..GRID {
            for i in 0..GRID {
                let x = i as f64 * (16.0 / GRID as f64);
                let y = j as f64 * (16.0 / GRID as f64);
                let dx = x - p.center[0];
                let dy = y - p.center[1];
                let r2 = dx * dx + dy * dy;
                let amp = p.amp * (-r2 * inv).exp();
                let phase = p.phase + p.k[0] * dx + p.k[1] * dy;
                self.psi[Self::ix(i, j)] += Complex64::from_polar(amp, phase);
            }
        }
    }

    /// Un paso Schrödinger lineal (sin |ψ|²): barato, solo dispersión.
    pub fn step_linear(&mut self) {
        let mut lap = vec![Complex64::new(0.0, 0.0); CELLS2];
        for j in 0..GRID {
            for i in 0..GRID {
                let c = self.psi[Self::ix(i, j)];
                let ip = Self::ix((i + 1) % GRID, j);
                let im = Self::ix((i + GRID - 1) % GRID, j);
                let jp = Self::ix(i, (j + 1) % GRID);
                let jm = Self::ix(i, (j + GRID - 1) % GRID);
                lap[Self::ix(i, j)] =
                    self.psi[ip] + self.psi[im] + self.psi[jp] + self.psi[jm] - c * 4.0;
            }
        }
        let i_unit = Complex64::new(0.0, 1.0);
        for (n, &l) in lap.iter().enumerate().take(CELLS2) {
            self.psi[n] += i_unit * self.alpha * l * self.dt;
        }
    }

    pub fn intensity_sum(&self) -> f64 {
        self.psi.iter().map(|z| z.norm_sqr()).sum()
    }

    /// Pico de |ψ|² (proxy de colapso / foco de interferencia).
    pub fn peak_intensity(&self) -> f64 {
        self.psi
            .iter()
            .map(|z| z.norm_sqr())
            .fold(0.0_f64, f64::max)
    }
}

impl Default for WaveField2D {
    fn default() -> Self {
        Self::new()
    }
}

/// Score en rejilla: past + future → evolve lineal → pico de intensidad.
pub fn grid_interference_score(past: &[WavePacket], future: &WavePacket, steps: usize) -> f64 {
    let mut field = WaveField2D::new();
    for p in past {
        field.add_packet(p);
    }
    let alone_past = field.peak_intensity();
    field.add_packet(future);
    for _ in 0..steps {
        field.step_linear();
    }
    let together = field.peak_intensity();
    // Exceso de pico respecto al pasado solo ≈ construcción.
    together - alone_past
}

pub fn predict_with_grid(
    past: &[WavePacket],
    future_contents: &[usize],
    steps: usize,
) -> PredictionReport {
    let mut best_i = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, &content) in future_contents.iter().enumerate() {
        let fut = WavePacket::future(i as u64, content, 1.0);
        let s = grid_interference_score(past, &fut, steps);
        if s > best_score {
            best_score = s;
            best_i = i;
        }
    }
    PredictionReport {
        best_future_id: best_i as u64,
        best_content: future_contents[best_i],
        best_score,
        scores: future_contents.len(),
    }
}

// ─── Bench ──────────────────────────────────────────────────────────────────

pub fn bench_analytic_predict(loops: usize, n_candidates: usize) -> (f64, f64) {
    let mut core = WavePredictCore::new();
    let mut ok = 0usize;
    let mut total = 0usize;
    let candidates: Vec<usize> = (0..n_candidates).collect();
    let t0 = Instant::now();
    for _ in 0..loops {
        for obs in 0..n_candidates {
            let r = core.predict_from_observation(obs, &candidates);
            total += 1;
            if r.best_content == obs {
                ok += 1;
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let us = (ms * 1e3) / total as f64;
    let acc = ok as f64 / total as f64;
    (acc, us)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_future_beats_mismatch() {
        let past = WavePacket::past(0, 3, 1.0);
        let good = WavePacket::future(1, 3, 1.0);
        let bad = WavePacket::future(2, 5, 1.0);
        let sg = analytic_interference(&past, &good);
        let sb = analytic_interference(&past, &bad);
        println!("match={sg:.4} mismatch={sb:.4}");
        assert!(sg > sb, "aligned future should construct more");
        assert!(sg > 0.0);
    }

    #[test]
    fn predict_picks_correct_continuation() {
        let mut core = WavePredictCore::new();
        let candidates = [0usize, 1, 2, 3, 4, 5, 6, 7];
        for obs in 0..8 {
            let r = core.predict_from_observation(obs, &candidates);
            println!(
                "obs={obs} pred={} score={:.4}",
                r.best_content, r.best_score
            );
            assert_eq!(r.best_content, obs, "future must match past content");
        }
    }

    #[test]
    fn multi_past_prefers_consistent_future() {
        let mut core = WavePredictCore::new();
        core.inject_past_content(2);
        core.inject_past_content(2);
        let r = core.predict_best_future(&[0, 2, 5, 7]);
        assert_eq!(r.best_content, 2);
    }

    #[test]
    fn grid_refine_agrees_on_winner() {
        let past = vec![WavePacket::past(0, 1, 1.0)];
        let cands = [0usize, 1, 4, 6];
        let analytic = {
            let mut core = WavePredictCore::new();
            core.past = past.clone();
            core.predict_best_future(&cands)
        };
        let grid = predict_with_grid(&past, &cands, 4);
        println!(
            "analytic winner={} grid winner={} (scores {:.3} vs {:.3})",
            analytic.best_content, grid.best_content, analytic.best_score, grid.best_score
        );
        assert_eq!(analytic.best_content, grid.best_content);
        assert_eq!(analytic.best_content, 1);
    }

    #[test]
    fn analytic_path_is_microsecond_scale() {
        let (acc, us) = bench_analytic_predict(40, 8);
        println!("analytic predict acc={acc:.3} µs/query={us:.3}");
        assert!(acc >= 0.99);
        assert!(
            us < 50.0,
            "analytic interference should stay ≪ spin NLS: {us} µs"
        );
    }
}
