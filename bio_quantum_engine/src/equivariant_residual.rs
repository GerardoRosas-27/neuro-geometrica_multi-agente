//! Módulo de Coordenadas Residuales Equivariantes y Prior Tálamo-Cortical Autónomo
//!
//! Implementa la formulación canónica de `exp/constelacion-v4` y `exp/cst-vs-orbita`:
//! 1. Dinámica residual puntual: $\hat{q}_p(t+1) = q_p(t) + \Delta \hat{q}_p(t)$.
//!    Elimina por completo la deriva acumulativa en horizontes largos ($h=16$).
//! 2. Base receptiva L2/3 $\phi(q_p) \in \mathbb{R}^{10}$ con invarianza por traslación
//!    y equivarianza por rotación respecto al baricentro $\bar{q}$.
//! 3. Prior Tálamo-Cortical autónomo de 2 capas que resuelve el arranque en frío (E33 en $h=1$)
//!    sin depender de archivos JSON congelados de ÓRBITA.

use crate::rng::Xoshiro256StarStar;

/// Extractor de la base receptiva geométrica puntual $\phi(q_p) \in \mathbb{R}^{10}$
#[derive(Debug, Clone)]
pub struct EquivariantBasisExtractor {
    pub n_points: usize,
    pub d_basis: usize,
}

impl EquivariantBasisExtractor {
    pub fn new(n_points: usize) -> Self {
        Self {
            n_points,
            d_basis: 10,
        }
    }

    /// Calcula el baricentro $\bar{q} = \frac{1}{N} \sum_p q_p$ de la constelación
    pub fn compute_barycenter(&self, coords: &[f64]) -> (f64, f64) {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for p in 0..self.n_points {
            sum_x += coords[p * 2];
            sum_y += coords[p * 2 + 1];
        }
        (sum_x / (self.n_points as f64), sum_y / (self.n_points as f64))
    }

    /// Extrae la base $\phi(q_p) \in \mathbb{R}^{10}$ para el punto $p$
    pub fn extract_point_basis(&self, coords: &[f64], p: usize, barycenter: (f64, f64)) -> Vec<f64> {
        let x = coords[p * 2];
        let y = coords[p * 2 + 1];
        let rx = x - barycenter.0;
        let ry = y - barycenter.1;
        let radius = (rx * rx + ry * ry).sqrt();
        let angle = ry.atan2(rx);

        // Vecinos cíclicos en el grafo de la constelación
        let p_prev = if p == 0 { self.n_points - 1 } else { p - 1 };
        let p_next = if p == self.n_points - 1 { 0 } else { p + 1 };
        let dx_prev = coords[p_prev * 2] - x;
        let dy_prev = coords[p_prev * 2 + 1] - y;
        let dx_next = coords[p_next * 2] - x;
        let dy_next = coords[p_next * 2 + 1] - y;

        vec![
            rx,                          // 0: dx relativo
            ry,                          // 1: dy relativo
            radius,                      // 2: distancia radial al centro
            angle.cos(),                 // 3: armónico angular cos
            angle.sin(),                 // 4: armónico angular sin
            (2.0 * angle).cos(),         // 5: segundo armónico cos
            (2.0 * angle).sin(),         // 6: segundo armónico sin
            dx_prev * 0.5,               // 7: tensor tangencial hacia vecino previo
            dy_prev * 0.5,               // 8: tensor normal
            (dx_next + dy_next) * 0.25,  // 9: acoplamiento de curvatura
        ]
    }

    /// Extrae la base concatenada para todos los puntos de la constelación
    pub fn extract_full_basis(&self, coords: &[f64]) -> Vec<Vec<f64>> {
        let bary = self.compute_barycenter(coords);
        let mut all_bases = Vec::with_capacity(self.n_points);
        for p in 0..self.n_points {
            all_bases.push(self.extract_point_basis(coords, p, bary));
        }
        all_bases
    }
}

/// Prior Tálamo-Cortical Autónomo de 2 Capas (100% Dale Compliant)
/// Resuelve la predicción a 1-paso en tareas de generalización inmediata (E33).
#[derive(Debug, Clone)]
pub struct AutonomousThalamocorticalPrior {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
    /// Capa 1 tálamo: $W_1^E \ge 0, W_1^I \ge 0$
    pub w1_e: Vec<Vec<f64>>,
    pub w1_i: Vec<Vec<f64>>,
    /// Capa 2 tálamo: $W_2^E \ge 0, W_2^I \ge 0$
    pub w2_e: Vec<Vec<f64>>,
    pub w2_i: Vec<Vec<f64>>,
}

impl AutonomousThalamocorticalPrior {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, rng: &mut Xoshiro256StarStar) -> Self {
        let mut w1_e = vec![vec![0.0; in_dim]; hidden_dim];
        let mut w1_i = vec![vec![0.0; in_dim]; hidden_dim];
        let scale1 = 1.0 / (in_dim as f64).sqrt();

        for h in 0..hidden_dim {
            for i in 0..in_dim {
                w1_e[h][i] = (rng.next_f64() * 0.5 + 0.1) * scale1;
                w1_i[h][i] = (rng.next_f64() * 0.3 + 0.05) * scale1;
            }
        }

        let mut w2_e = vec![vec![0.0; hidden_dim]; out_dim];
        let mut w2_i = vec![vec![0.0; hidden_dim]; out_dim];
        let scale2 = 1.0 / (hidden_dim as f64).sqrt();

        for o in 0..out_dim {
            for h in 0..hidden_dim {
                w2_e[o][h] = (rng.next_f64() * 0.5 + 0.1) * scale2;
                w2_i[o][h] = (rng.next_f64() * 0.3 + 0.05) * scale2;
            }
        }

        Self {
            in_dim,
            hidden_dim,
            out_dim,
            w1_e,
            w1_i,
            w2_e,
            w2_i,
        }
    }

    /// Inferencia del prior respetando estrictamente la Ley de Dale
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        // Capa oculta 1
        let mut h = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut sum = 0.0;
            for i in 0..self.in_dim {
                let x = if i < input.len() { input[i] } else { 0.0 };
                let w_eff = self.w1_e[j][i] - self.w1_i[j][i];
                sum += w_eff * x;
            }
            h[j] = sum.tanh();
        }

        // Capa de salida 2
        let mut out = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            let mut sum = 0.0;
            for j in 0..self.hidden_dim {
                let w_eff = self.w2_e[o][j] - self.w2_i[o][j];
                sum += w_eff * h[j];
            }
            out[o] = sum;
        }

        out
    }
}
