//! Conector 4: Constelación v4 & ÓRBITA -> Jerarquía Tipo-Token Marcus.
//!
//! Integra la memoria de constelación del repositorio:
//! - Memoria estructurada de prototipos categóricos (Constelación v4 / ÓRBITA).
//! - Mapeo riguroso de nodos de la constelación a Tipos Ontológicos Abstractos.
//! - Instanciación de Tokens espacio-temporales episódicos que heredan las restricciones del Tipo.
//! - Evita la falacia de inducción superficial denunciada por Marcus.

use crate::algebraic_mind::type_token::{AbstractType, TokenId, TokenInstance, TypeId, TypeTokenRegistry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Nodo prototípico en la memoria de la Constelación v4.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstellationNode {
    pub node_id: usize,
    pub label: String,
    pub parent_node_id: Option<usize>,
    pub is_solid: bool,
    pub is_container: bool,
    pub is_agent: bool,
    pub default_mass: f64,
}

/// Conector que traduce la topología de la Constelación v4 a la taxonomía Tipo-Token de Marcus.
#[derive(Debug, Clone, Default)]
pub struct ConstellationTypeTokenConnector {
    nodes: HashMap<usize, ConstellationNode>,
    node_to_type_id: HashMap<usize, TypeId>,
}

impl ConstellationTypeTokenConnector {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            node_to_type_id: HashMap::new(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Registra un nodo de la Constelación v4 y lo convierte automáticamente en un Tipo ontológico en el registro de Marcus.
    pub fn ingest_constellation_node(
        &mut self,
        node: ConstellationNode,
        registry: &mut TypeTokenRegistry,
    ) {
        let type_id = TypeId::new(format!("Constellation_{}", node.label));

        let parent_type = node.parent_node_id.and_then(|p_id| self.node_to_type_id.get(&p_id).cloned());

        let abstract_type = AbstractType {
            id: type_id.clone(),
            parent_type,
            is_solid: node.is_solid,
            is_container: node.is_container,
            is_intentional_agent: node.is_agent,
            subject_to_gravity: true,
        };

        registry.register_type(abstract_type);

        self.node_to_type_id.insert(node.node_id, type_id);
        self.nodes.insert(node.node_id, node);
    }

    /// Instancia un Token episódico a partir de un nodo prototípico de la Constelación.
    pub fn spawn_token_from_constellation(
        &self,
        node_id: usize,
        token_label: impl Into<String>,
        position: [f64; 3],
        registry: &mut TypeTokenRegistry,
    ) -> Result<TokenId, String> {
        let type_id = self.node_to_type_id.get(&node_id).ok_or_else(|| {
            format!("Nodo de constelación {} no registrado", node_id)
        })?;

        let node = self.nodes.get(&node_id).unwrap();
        let token_id = TokenId::random_uuid();

        let mut token = TokenInstance::new(
            token_id.0.clone(),
            type_id.0.clone(),
            token_label,
            position,
        );
        token.mass = node.default_mass;

        registry.register_token(token)?;
        Ok(token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constellation_to_type_token_bridge() {
        let mut registry = TypeTokenRegistry::new();
        let mut bridge = ConstellationTypeTokenConnector::new();

        // 1. Ingestar nodos de la Constelación v4
        let parent_node = ConstellationNode {
            node_id: 1,
            label: "Vehicle".to_string(),
            parent_node_id: None,
            is_solid: true,
            is_container: true,
            is_agent: false,
            default_mass: 1500.0,
        };
        bridge.ingest_constellation_node(parent_node, &mut registry);

        let child_node = ConstellationNode {
            node_id: 2,
            label: "AutonomousRover".to_string(),
            parent_node_id: Some(1),
            is_solid: true,
            is_container: true,
            is_agent: true,
            default_mass: 250.0,
        };
        bridge.ingest_constellation_node(child_node, &mut registry);

        // 2. Instanciar token episódico
        let token_id = bridge
            .spawn_token_from_constellation(2, "Curiosity_Rover_Unit_1", [10.0, 5.0, 0.0], &mut registry)
            .expect("Debe instanciar el token exitosamente");

        let token = registry.get_token(&token_id).unwrap();
        assert_eq!(token.label, "Curiosity_Rover_Unit_1");
        assert_eq!(token.mass, 250.0);

        // Verificar que hereda las propiedades del tipo
        let token_type = registry.get_type(&token.type_id).unwrap();
        assert!(token_type.is_solid);
        assert!(token_type.is_intentional_agent);
    }
}
