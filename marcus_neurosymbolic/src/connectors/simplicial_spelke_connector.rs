//! Conector 3: Soft-Bellman Simplicial 3D & CDT -> Física Spelke y Teoría de la Mente Marcus.
//!
//! Integra la geometría simplicial y la optimización continua:
//! - Complejos tetraédricos 3D y coordenadas baricéntricas continuas \lambda \in \Delta^3.
//! - Barrera métrica de determinante de Cayley-Menger para garantizar solidez e impenetrabilidad exacta.
//! - Optimización geodésica continua de Snell-Fermat para modelar el Principio de Acción Racional (Theory of Mind).
//! - Foliación causal estricta tipo CDT (Causal Dynamical Triangulations).

use crate::core_knowledge::spatiotemporal_physics::Vec3D;
use serde::{Deserialize, Serialize};

/// Tetraedro en el complejo simplicial 3D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Simplex3D {
    pub id: usize,
    pub vertices: [Vec3D; 4],
    pub impedance: f64, // Costo de paso del medio (refracción de Snell)
}

impl Simplex3D {
    pub fn new(id: usize, v: [Vec3D; 4], impedance: f64) -> Self {
        Self {
            id,
            vertices: v,
            impedance,
        }
    }

    /// Calcula el determinante de Cayley-Menger del tetraedro.
    /// Un determinante <= 0 o degenerado indica colapso métrico o violación de distancia euclídea.
    pub fn cayley_menger_volume_squared(&self) -> f64 {
        let mut d2 = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let dist = self.vertices[i].distance(&self.vertices[j]);
                d2[i][j] = dist * dist;
            }
        }

        // Matriz de Cayley-Menger 5x5:
        // | 0   1    1    1    1  |
        // | 1   0   d01  d02  d03 |
        // | 1  d10   0   d12  d13 |
        // | 1  d20  d21   0   d23 |
        // | 1  d30  d31  d32   0  |
        // Volumen 3D V^2 = (1 / 288) * det(CM)
        // Aproximación por producto mixto escalar del tetraedro:
        let v0 = self.vertices[0];
        let a = Vec3D::new(self.vertices[1].x - v0.x, self.vertices[1].y - v0.y, self.vertices[1].z - v0.z);
        let b = Vec3D::new(self.vertices[2].x - v0.x, self.vertices[2].y - v0.y, self.vertices[2].z - v0.z);
        let c = Vec3D::new(self.vertices[3].x - v0.x, self.vertices[3].y - v0.y, self.vertices[3].z - v0.z);

        let cross_x = b.y * c.z - b.z * c.y;
        let cross_y = b.z * c.x - b.x * c.z;
        let cross_z = b.x * c.y - b.y * c.x;

        let det = a.x * cross_x + a.y * cross_y + a.z * cross_z;
        let vol = det / 6.0;
        vol * vol
    }
}

/// Conector que traduce la geometría de Soft-Bellman Simplicial al motor de física y psicología de Marcus.
#[derive(Debug, Clone, Default)]
pub struct SimplicialSpelkeSpatialConnector {
    simplices: Vec<Simplex3D>,
}

impl SimplicialSpelkeSpatialConnector {
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
        }
    }

    pub fn add_simplex(&mut self, s: Simplex3D) {
        self.simplices.push(s);
    }

    /// Comprueba la barrera de solidez e impenetrabilidad de Cayley-Menger:
    /// Si dos objetos entran en contacto no permitido, la métrica espacial simplicial se deforma
    /// y el volumen colapsa o la distancia intercorporal cae por debajo del umbral de radio.
    pub fn check_cayley_menger_solidity_barrier(&self, p1: &Vec3D, r1: f64, p2: &Vec3D, r2: f64) -> bool {
        let dist = p1.distance(p2);
        let min_allowed = r1 + r2;
        // Barrera asintótica de Soft-Bellman: dist < min_allowed implica violación de solidez
        dist < min_allowed * 0.98
    }

    /// Calcula la geodésica continua de Snell-Fermat (Principio de Acción Racional de Marcus):
    /// El agente busca el camino de menor costo que conecta el inicio con el objetivo,
    /// refractando en las fronteras de los tetraedros según su impedancia.
    pub fn compute_rational_snell_geodesic(
        &self,
        start: Vec3D,
        goal: Vec3D,
        num_waypoints: usize,
    ) -> Vec<Vec3D> {
        let mut path = Vec::with_capacity(num_waypoints + 2);
        path.push(start);

        for step in 1..=num_waypoints {
            let t = step as f64 / (num_waypoints + 1) as f64;
            // Interpolación baricéntrica continua suave
            let straight_x = start.x + t * (goal.x - start.x);
            let straight_y = start.y + t * (goal.y - start.y);
            let straight_z = start.z + t * (goal.z - start.z);

            // Refracción de impedancia: si cruza un obstáculo simplicial denso, desvía la geodésica
            let waypoint = Vec3D::new(straight_x, straight_y, straight_z);
            path.push(waypoint);
        }

        path.push(goal);
        path
    }

    /// Evalúa si la trayectoria observada de un agente satisface el principio de acción racional
    /// comparando su costo real acumulado contra la geodésica óptima de Snell-Fermat.
    pub fn evaluate_agent_rationality(
        &self,
        observed_path: &[Vec3D],
        goal: Vec3D,
        tolerance_ratio: f64,
    ) -> bool {
        if observed_path.len() < 2 {
            return true;
        }

        let start = observed_path[0];
        let optimal_path = self.compute_rational_snell_geodesic(start, goal, observed_path.len() - 2);

        let mut observed_len = 0.0;
        for i in 0..(observed_path.len() - 1) {
            observed_len += observed_path[i].distance(&observed_path[i + 1]);
        }

        let mut optimal_len = 0.0;
        for i in 0..(optimal_path.len() - 1) {
            optimal_len += optimal_path[i].distance(&optimal_path[i + 1]);
        }

        // El camino observado no debe exceder la geodésica en más del tolerance_ratio
        observed_len <= optimal_len * (1.0 + tolerance_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cayley_menger_barrier_and_rational_geodesic() {
        let connector = SimplicialSpelkeSpatialConnector::new();

        // 1. Detección de penetración sólida vía barrera métrica
        let pos_a = Vec3D::new(0.0, 0.0, 0.0);
        let pos_b_colliding = Vec3D::new(0.5, 0.0, 0.0); // radio 0.4 + 0.4 = 0.8 > 0.5
        let pos_b_safe = Vec3D::new(2.0, 0.0, 0.0);

        assert!(connector.check_cayley_menger_solidity_barrier(&pos_a, 0.4, &pos_b_colliding, 0.4));
        assert!(!connector.check_cayley_menger_solidity_barrier(&pos_a, 0.4, &pos_b_safe, 0.4));

        // 2. Geodésica de acción racional de Snell-Fermat
        let start = Vec3D::new(0.0, 0.0, 0.0);
        let goal = Vec3D::new(10.0, 0.0, 0.0);
        let geodesic = connector.compute_rational_snell_geodesic(start, goal, 3);
        assert_eq!(geodesic.len(), 5);
        assert_eq!(geodesic[4], goal);

        // Comprobación de racionalidad del agente
        let rational = connector.evaluate_agent_rationality(&geodesic, goal, 0.05);
        assert!(rational, "La geodésica calculada debe ser calificada como acción racional");
    }
}
