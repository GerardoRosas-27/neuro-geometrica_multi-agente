//! Módulo de Psicología Núcleo y Agencia Intencional (Theory of Mind).
//!
//! Basado en Gary Marcus (The Birth of the Mind, 2004 y Rebooting AI, 2019):
//! "Infants differentiate between inanimate objects and intentional agents.
//! Agents are understood to possess goals, perceptual gaze, and to act rationally
//! by choosing efficient paths to reach their objectives."

use crate::core_knowledge::spatiotemporal_physics::Vec3D;
use serde::{Deserialize, Serialize};

/// Meta u objetivo de un agente intencional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentGoal {
    pub target_position: Vec3D,
    pub tolerance: f64,
    pub description: String,
}

/// Estado intencional de un agente.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveAgent {
    pub id: String,
    pub current_position: Vec3D,
    pub current_goal: Option<AgentGoal>,
    /// Trayectoria reciente recorrida por el agente.
    pub trajectory: Vec<Vec3D>,
}

impl CognitiveAgent {
    pub fn new(id: impl Into<String>, start_pos: Vec3D) -> Self {
        Self {
            id: id.into(),
            current_position: start_pos,
            current_goal: None,
            trajectory: vec![start_pos],
        }
    }

    pub fn set_goal(&mut self, goal: AgentGoal) {
        self.current_goal = Some(goal);
    }

    /// Evalúa el principio de acción racional (Gergely & Csibra / Marcus):
    /// Un agente racional debe reducir la distancia a su meta a lo largo de su trayectoria
    /// a menos que un obstáculo físico imponga una desviación geodésica.
    pub fn is_rationally_pursuing_goal(&self) -> bool {
        if let Some(goal) = &self.current_goal {
            if self.trajectory.len() < 2 {
                return true;
            }
            let initial_dist = self.trajectory.first().unwrap().distance(&goal.target_position);
            let current_dist = self.current_position.distance(&goal.target_position);
            // El agente ha progresado hacia la meta o ya está en la tolerancia
            current_dist <= initial_dist || current_dist <= goal.tolerance
        } else {
            // Sin meta explícita no se juzga racionalidad teleológica
            true
        }
    }

    /// Da un paso hacia la meta con una velocidad y dirección dadas.
    pub fn step_towards(&mut self, target: Vec3D, step_size: f64) {
        let dist = self.current_position.distance(&target);
        if dist > 1e-6 {
            let dx = (target.x - self.current_position.x) / dist * step_size.min(dist);
            let dy = (target.y - self.current_position.y) / dist * step_size.min(dist);
            let dz = (target.z - self.current_position.z) / dist * step_size.min(dist);
            self.current_position.x += dx;
            self.current_position.y += dy;
            self.current_position.z += dz;
            self.trajectory.push(self.current_position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_action_principle() {
        let mut agent = CognitiveAgent::new("Agent1", Vec3D::new(0.0, 0.0, 0.0));
        let goal = AgentGoal {
            target_position: Vec3D::new(10.0, 0.0, 0.0),
            tolerance: 0.1,
            description: "Reach food source".to_string(),
        };
        agent.set_goal(goal.clone());

        // Da 3 pasos hacia la meta
        agent.step_towards(goal.target_position, 2.0);
        agent.step_towards(goal.target_position, 2.0);

        assert!(agent.is_rationally_pursuing_goal());
        assert_eq!(agent.current_position.x, 4.0);

        // Si el agente ahora da pasos en dirección diametralmente opuesta alejándose
        agent.step_towards(Vec3D::new(-20.0, 0.0, 0.0), 10.0);
        // Ahora está en -6.0, más lejos del objetivo inicial (10.0) que al principio (dist = 16 > 10)
        assert!(!agent.is_rationally_pursuing_goal());
    }
}
