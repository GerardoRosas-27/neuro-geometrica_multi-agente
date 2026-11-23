//! Benchmark 3: Conservación Ontológica y Física Núcleo de Spelke.
//!
//! Evalúa la hipótesis H3 de Gary Marcus (The Birth of the Mind, 2004 y Rebooting AI, 2019):
//! "Common-sense physical understanding cannot be reliably induced purely from surface statistics.
//! An innate core physics engine deterministically catches solidity, permanence, and continuity violations."

use crate::core_knowledge::spatiotemporal_physics::{
    PhysicalViolation, SpatiotemporalObject, SpelkePhysicsEngine, Vec3D,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm3Results {
    pub name: String,
    pub total_scenarios: usize,
    pub violation_recall: f64,
    pub false_positive_rate: f64,
    pub passed_h3: bool,
}

pub fn run_bm3_spelke_physics() -> Bm3Results {
    let mut engine = SpelkePhysicsEngine::new();
    engine.max_physical_speed = 30.0;

    let mut true_violations = 0;
    let mut detected_violations = 0;
    let mut valid_scenarios = 0;
    let mut false_positives = 0;

    // 1. Escenarios Válidos (Movimiento continuo y reposo a lo largo de 50 pasos de tiempo)
    for i in 0..50 {
        valid_scenarios += 1;
        let t = i as f64 * 0.1;
        let obj = SpatiotemporalObject {
            id: "continuous_traveler".to_string(),
            position: Vec3D::new(t * 2.0, 0.0, 0.0), // velocidad 2 m/s (< 30 m/s)
            velocity: Vec3D::new(2.0, 0.0, 0.0),
            radius: 0.3,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };
        let viols = engine.step_and_verify(&[obj], 0.1);
        if !viols.is_empty() {
            false_positives += 1;
        }
    }

    // 2. Escenarios con Violación de Teleportación (20 casos)
    for i in 0..20 {
        true_violations += 1;
        let mut local_engine = SpelkePhysicsEngine::new();
        local_engine.max_physical_speed = 20.0;
        let obj_t0 = SpatiotemporalObject {
            id: format!("teleport_obj_{}", i),
            position: Vec3D::new(0.0, 0.0, 0.0),
            velocity: Vec3D::zero(),
            radius: 0.4,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };
        local_engine.step_and_verify(&[obj_t0], 0.1);

        // Salto discontinuo de 200 metros en 0.1s (2000 m/s > 20 m/s)
        let obj_t1 = SpatiotemporalObject {
            id: format!("teleport_obj_{}", i),
            position: Vec3D::new(200.0, 0.0, 0.0),
            velocity: Vec3D::zero(),
            radius: 0.4,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };
        let viols = local_engine.step_and_verify(&[obj_t1], 0.1);
        if viols.iter().any(|v| matches!(v, PhysicalViolation::Teleportation { .. })) {
            detected_violations += 1;
        }
    }

    // 3. Escenarios con Violación de Solidez / Penetración (20 casos)
    for i in 0..20 {
        true_violations += 1;
        let mut local_engine = SpelkePhysicsEngine::new();
        let obj_a = SpatiotemporalObject {
            id: format!("solid_a_{}", i),
            position: Vec3D::new(0.0, 0.0, 0.0),
            velocity: Vec3D::zero(),
            radius: 0.5,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };
        // Intersección directa (distancia 0.1 < 0.5 + 0.5 = 1.0)
        let obj_b = SpatiotemporalObject {
            id: format!("solid_b_{}", i),
            position: Vec3D::new(0.1, 0.0, 0.0),
            velocity: Vec3D::zero(),
            radius: 0.5,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };
        let viols = local_engine.step_and_verify(&[obj_a, obj_b], 0.1);
        if viols.iter().any(|v| matches!(v, PhysicalViolation::SolidInterpenetration { .. })) {
            detected_violations += 1;
        }
    }

    // 4. Escenarios con Violación de Permanencia de Objeto (10 casos)
    for i in 0..10 {
        true_violations += 1;
        let mut local_engine = SpelkePhysicsEngine::new();
        let obj = SpatiotemporalObject {
            id: format!("permanence_obj_{}", i),
            position: Vec3D::new(5.0, 5.0, 0.0),
            velocity: Vec3D::zero(),
            radius: 0.5,
            is_solid: true,
            is_occluded: false,
            is_supported: true,
            contained_in: None,
        };
        local_engine.step_and_verify(&[obj], 0.1);

        // En el siguiente paso el objeto desaparece mágicamente de la simulación
        let viols = local_engine.step_and_verify(&[], 0.1);
        if viols.iter().any(|v| matches!(v, PhysicalViolation::ObjectPermanenceFailure { .. })) {
            detected_violations += 1;
        }
    }

    let recall = detected_violations as f64 / true_violations as f64;
    let fpr = false_positives as f64 / valid_scenarios as f64;

    Bm3Results {
        name: "BM-3: Física Núcleo y Permanencia Spelke".to_string(),
        total_scenarios: valid_scenarios + true_violations,
        violation_recall: recall,
        false_positive_rate: fpr,
        passed_h3: recall >= 0.99 && fpr <= 0.01,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm3_execution() {
        let res = run_bm3_spelke_physics();
        assert!(res.passed_h3);
        assert_eq!(res.violation_recall, 1.0);
        assert_eq!(res.false_positive_rate, 0.0);
    }
}
