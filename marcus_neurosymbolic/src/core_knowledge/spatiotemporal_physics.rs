//! Módulo de Física Núcleo Espacio-Temporal (Spelke Core Physics).
//!
//! Basado en Gary Marcus (The Birth of the Mind, 2004 y Rebooting AI, 2019):
//! "Humans are born with innate cognitive scaffolds (Spelke core knowledge):
//! spatiotemporal continuity, solidity, gravity, and object permanence."

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Vector tridimensional euclídeo.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Estado de un objeto físico en un instante temporal $t$.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatiotemporalObject {
    pub id: String,
    pub position: Vec3D,
    pub velocity: Vec3D,
    pub radius: f64,
    pub is_solid: bool,
    pub is_occluded: bool,
    pub is_supported: bool,
    pub contained_in: Option<String>,
}

impl SpatiotemporalObject {
    pub fn new(id: impl Into<String>, pos: (f64, f64, f64), radius: f64) -> Self {
        Self {
            id: id.into(),
            position: Vec3D::new(pos.0, pos.1, pos.2),
            velocity: Vec3D::zero(),
            radius,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        }
    }
}

/// Violaciones físicas detectadas por el motor de conocimiento núcleo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhysicalViolation {
    /// Teleportación: El objeto cambió de posición violando la continuidad espacial.
    Teleportation {
        object_id: String,
        dt: f64,
        observed_speed: f64,
        max_speed: f64,
    },
    /// Penetración: Dos cuerpos rígidos sólidos se intersectan.
    SolidInterpenetration {
        object_a: String,
        object_b: String,
        distance: f64,
        min_allowed_distance: f64,
    },
    /// Desaparición Mágica: Violación de permanencia de objeto (desaparece sin justificación causal).
    ObjectPermanenceFailure {
        object_id: String,
        last_position: Vec3D,
    },
    /// Violación de Gravedad: Objeto en el aire sin soporte y sin velocidad ascendente que no cae.
    GravityDefiance {
        object_id: String,
        altitude_z: f64,
    },
    /// Violación de Contención Transitiva: A está en B, B está en C, pero A se declara fuera de C.
    ContainmentTransitivityFailure {
        object_a: String,
        container_b: String,
        container_c: String,
    },
}

/// Motor de simulación y verificación física de sentido común.
#[derive(Debug, Clone)]
pub struct SpelkePhysicsEngine {
    pub max_physical_speed: f64,
    pub gravity_acceleration: f64,
    /// Historial de objetos en el tiempo anterior (t-1).
    previous_state: HashMap<String, SpatiotemporalObject>,
    /// Relaciones de contención: Inside(a, b).
    containment_graph: HashMap<String, String>, // child -> parent
}

impl SpelkePhysicsEngine {
    pub fn new() -> Self {
        Self {
            max_physical_speed: 100.0, // m/s límite físico razonable para objetos macroscópicos cotidianos
            gravity_acceleration: 9.81,
            previous_state: HashMap::new(),
            containment_graph: HashMap::new(),
        }
    }

    /// Registra la relación: objeto `child` está dentro de `parent`.
    pub fn set_containment(&mut self, child: impl Into<String>, parent: impl Into<String>) {
        self.containment_graph.insert(child.into(), parent.into());
    }

    /// Verifica si `child` está dentro de `target_parent` considerando la clausura transitiva.
    pub fn is_transitively_inside(&self, child: &str, target_parent: &str) -> bool {
        let mut current = child;
        let mut visited = HashSet::new();

        while let Some(parent) = self.containment_graph.get(current) {
            if parent == target_parent {
                return true;
            }
            if !visited.insert(parent.as_str()) {
                break; // Ciclo defensivo
            }
            current = parent;
        }
        false
    }

    /// Evalúa una transición de estado en el intervalo dt y retorna todas las violaciones detectadas.
    pub fn step_and_verify(
        &mut self,
        current_objects: &[SpatiotemporalObject],
        dt: f64,
    ) -> Vec<PhysicalViolation> {
        let mut violations = Vec::new();
        let mut current_map = HashMap::new();

        for obj in current_objects {
            current_map.insert(obj.id.clone(), obj.clone());

            // 1. Verificar Continuidad Espacio-Temporal con el estado previo
            if let Some(prev) = self.previous_state.get(&obj.id) {
                let dist = prev.position.distance(&obj.position);
                let speed = dist / dt.max(1e-6);
                if speed > self.max_physical_speed {
                    violations.push(PhysicalViolation::Teleportation {
                        object_id: obj.id.clone(),
                        dt,
                        observed_speed: speed,
                        max_speed: self.max_physical_speed,
                    });
                }
            }

            // 2. Verificar Gravedad / Soporte
            if obj.position.z > 0.05 && !obj.is_supported && obj.contained_in.is_none() {
                // Si no tiene aceleración hacia abajo o soporte, es anomalía
                if obj.velocity.z >= 0.0 {
                    violations.push(PhysicalViolation::GravityDefiance {
                        object_id: obj.id.clone(),
                        altitude_z: obj.position.z,
                    });
                }
            }
        }

        // 3. Verificar Solidez / Impenetrabilidad entre todos los pares
        for i in 0..current_objects.len() {
            for j in (i + 1)..current_objects.len() {
                let a = &current_objects[i];
                let b = &current_objects[j];

                // Si uno contiene al otro, no hay colisión externa
                if a.contained_in.as_ref() == Some(&b.id) || b.contained_in.as_ref() == Some(&a.id) {
                    continue;
                }

                if a.is_solid && b.is_solid {
                    let dist = a.position.distance(&b.position);
                    let min_dist = a.radius + b.radius;
                    if dist < min_dist * 0.95 {
                        violations.push(PhysicalViolation::SolidInterpenetration {
                            object_a: a.id.clone(),
                            object_b: b.id.clone(),
                            distance: dist,
                            min_allowed_distance: min_dist,
                        });
                    }
                }
            }
        }

        // 4. Verificar Permanencia de Objeto:
        // Si un objeto estaba en prev_state y no está en current, pero no fue destruido ni consumido
        for (prev_id, prev_obj) in &self.previous_state {
            if !current_map.contains_key(prev_id) {
                // Desaparición mágica violando permanencia
                violations.push(PhysicalViolation::ObjectPermanenceFailure {
                    object_id: prev_id.clone(),
                    last_position: prev_obj.position,
                });
            }
        }

        self.previous_state = current_map;
        violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teleportation_detection() {
        let mut engine = SpelkePhysicsEngine::new();
        engine.max_physical_speed = 50.0;

        let obj_t0 = SpatiotemporalObject {
            id: "Ball".to_string(),
            position: Vec3D::new(0.0, 0.0, 0.0),
            velocity: Vec3D::zero(),
            radius: 0.2,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };

        // Paso inicial
        let v0 = engine.step_and_verify(&[obj_t0], 0.1);
        assert!(v0.is_empty());

        // En dt = 0.1s se mueve 500 metros (velocidad = 5000 m/s > 50 m/s) -> Teleportación
        let obj_t1 = SpatiotemporalObject {
            id: "Ball".to_string(),
            position: Vec3D::new(500.0, 0.0, 0.0),
            velocity: Vec3D::zero(),
            radius: 0.2,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };

        let v1 = engine.step_and_verify(&[obj_t1], 0.1);
        assert_eq!(v1.len(), 1);
        match &v1[0] {
            PhysicalViolation::Teleportation { object_id, .. } => {
                assert_eq!(object_id, "Ball");
            }
            _ => panic!("Esperaba violación de teleportación"),
        }
    }

    #[test]
    fn test_solidity_interpenetration() {
        let mut engine = SpelkePhysicsEngine::new();

        let obj_a = SpatiotemporalObject {
            id: "BlockA".to_string(),
            position: Vec3D::new(1.0, 0.0, 0.0),
            velocity: Vec3D::zero(),
            radius: 0.5,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };

        // BlockB se coloca a 0.2m (distancia requerida: 0.5 + 0.5 = 1.0m)
        let obj_b = SpatiotemporalObject {
            id: "BlockB".to_string(),
            position: Vec3D::new(1.2, 0.0, 0.0),
            velocity: Vec3D::zero(),
            radius: 0.5,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };

        let v = engine.step_and_verify(&[obj_a, obj_b], 0.1);
        assert!(v.iter().any(|viol| matches!(viol, PhysicalViolation::SolidInterpenetration { .. })));
    }

    #[test]
    fn test_containment_transitivity() {
        let mut engine = SpelkePhysicsEngine::new();
        // Lápiz en Taza, Taza en Mochila
        engine.set_containment("Pencil", "Cup");
        engine.set_containment("Cup", "Backpack");

        // Marcus: El lápiz debe estar transitivamente en la Mochila
        assert!(engine.is_transitively_inside("Pencil", "Backpack"));
        assert!(engine.is_transitively_inside("Pencil", "Cup"));
        assert!(!engine.is_transitively_inside("Backpack", "Pencil"));
    }
}
