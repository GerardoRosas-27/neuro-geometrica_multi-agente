//! Módulo de Distinción Ontológica Tipo-Token (Types vs Tokens).
//!
//! Basado en Gary Marcus (The Algebraic Mind, Cap. 2 y Rebooting AI, Cap. 5):
//! "A robust cognitive architecture must clearly separate abstract kinds (types, which possess
//! universal laws and affordances) from individual instances situated in spacetime (tokens, which
//! possess transient episodic coordinates and physical states)."

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identificador de tipo ontológico abstracto.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeId(pub String);

impl TypeId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// Identificador único de un token espacio-temporal concreto.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenId(pub String);

impl TokenId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn random_uuid() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

/// Propiedades invariantes del Tipo Ontológico.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractType {
    pub id: TypeId,
    pub parent_type: Option<TypeId>,
    /// Si los cuerpos de este tipo son impenetrables (sólidos).
    pub is_solid: bool,
    /// Si este tipo puede contener a otros cuerpos dentro.
    pub is_container: bool,
    /// Si este tipo posee agencia e intencionalidad propia (metas y propósitos).
    pub is_intentional_agent: bool,
    /// Si este tipo es afectado por la gravedad newtoniana.
    pub subject_to_gravity: bool,
}

impl AbstractType {
    pub fn new_rigid_solid(id: impl Into<String>) -> Self {
        Self {
            id: TypeId::new(id),
            parent_type: None,
            is_solid: true,
            is_container: false,
            is_intentional_agent: false,
            subject_to_gravity: true,
        }
    }

    pub fn new_container(id: impl Into<String>) -> Self {
        Self {
            id: TypeId::new(id),
            parent_type: Some(TypeId::new("RigidSolid")),
            is_solid: true,
            is_container: true,
            is_intentional_agent: false,
            subject_to_gravity: true,
        }
    }

    pub fn new_agent(id: impl Into<String>) -> Self {
        Self {
            id: TypeId::new(id),
            parent_type: None,
            is_solid: true,
            is_container: false,
            is_intentional_agent: true,
            subject_to_gravity: true,
        }
    }
}

/// Estado episódico situado de un Token espacio-temporal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInstance {
    pub id: TokenId,
    pub type_id: TypeId,
    pub label: String,
    pub position: [f64; 3], // (x, y, z)
    pub velocity: [f64; 3], // (vx, vy, vz)
    pub radius: f64,        // radio volumétrico aproximado
    pub mass: f64,
    pub is_occluded: bool,  // Si está oculto tras un obstáculo
    pub contained_in: Option<TokenId>, // Si está dentro de otro token
}

impl TokenInstance {
    pub fn new(id: impl Into<String>, type_id: impl Into<String>, label: impl Into<String>, pos: [f64; 3]) -> Self {
        Self {
            id: TokenId::new(id),
            type_id: TypeId::new(type_id),
            label: label.into(),
            position: pos,
            velocity: [0.0, 0.0, 0.0],
            radius: 0.5,
            mass: 1.0,
            is_occluded: false,
            contained_in: None,
        }
    }

    pub fn euclidean_distance(&self, other: &Self) -> f64 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Registro unificado de Tipos y Tokens.
#[derive(Debug, Clone, Default)]
pub struct TypeTokenRegistry {
    types: HashMap<TypeId, AbstractType>,
    tokens: HashMap<TokenId, TokenInstance>,
}

impl TypeTokenRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            types: HashMap::new(),
            tokens: HashMap::new(),
        };
        // Inicializar tipos ontológicos base
        reg.register_type(AbstractType::new_rigid_solid("RigidSolid"));
        reg.register_type(AbstractType::new_container("BoxContainer"));
        reg.register_type(AbstractType::new_agent("CognitiveAgent"));
        reg
    }

    pub fn register_type(&mut self, t: AbstractType) {
        self.types.insert(t.id.clone(), t);
    }

    pub fn get_type(&self, id: &TypeId) -> Option<&AbstractType> {
        self.types.get(id)
    }

    pub fn register_token(&mut self, token: TokenInstance) -> Result<(), String> {
        if !self.types.contains_key(&token.type_id) {
            return Err(format!(
                "Error ontológico: tipo '{}' no registrado en la taxonomía",
                token.type_id.0
            ));
        }
        self.tokens.insert(token.id.clone(), token);
        Ok(())
    }

    pub fn get_token(&self, id: &TokenId) -> Option<&TokenInstance> {
        self.tokens.get(id)
    }

    pub fn get_token_mut(&mut self, id: &TokenId) -> Option<&mut TokenInstance> {
        self.tokens.get_mut(id)
    }

    pub fn all_tokens(&self) -> impl Iterator<Item = &TokenInstance> {
        self.tokens.values()
    }

    /// Comprueba si dos tokens violan el principio de solidez e impenetrabilidad:
    /// Dos cuerpos sólidos no pueden penetrarse ocupando el mismo espacio físico.
    pub fn check_solidity_violation(&self, id1: &TokenId, id2: &TokenId) -> bool {
        if let (Some(tok1), Some(tok2)) = (self.tokens.get(id1), self.tokens.get(id2)) {
            // Si uno está contenido en el otro de forma legítima, no es violación de solidez exterior
            if tok1.contained_in.as_ref() == Some(&tok2.id) || tok2.contained_in.as_ref() == Some(&tok1.id) {
                return false;
            }
            let type1 = self.types.get(&tok1.type_id);
            let type2 = self.types.get(&tok2.type_id);

            let solid1 = type1.map(|t| t.is_solid).unwrap_or(true);
            let solid2 = type2.map(|t| t.is_solid).unwrap_or(true);

            if solid1 && solid2 {
                let dist = tok1.euclidean_distance(tok2);
                let min_allowed = tok1.radius + tok2.radius;
                return dist < min_allowed * 0.95; // 5% de tolerancia numérica
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_token_separation_and_solidity() {
        let mut reg = TypeTokenRegistry::new();

        let token_cup = TokenInstance::new("cup_1", "RigidSolid", "Blue Cup", [0.0, 0.0, 1.0]);
        let token_stone = TokenInstance::new("stone_1", "RigidSolid", "Granite", [0.0, 0.0, 1.0]); // Misma posición

        assert!(reg.register_token(token_cup).is_ok());
        assert!(reg.register_token(token_stone).is_ok());

        // Dos sólidos en la misma posición [0, 0, 1] violan impenetrabilidad
        let violation = reg.check_solidity_violation(&TokenId::new("cup_1"), &TokenId::new("stone_1"));
        assert!(violation, "Debe detectar violación de solidez e impenetrabilidad");

        // Si movemos la piedra a 5 unidades de distancia, la violación desaparece
        if let Some(stone) = reg.get_token_mut(&TokenId::new("stone_1")) {
            stone.position = [5.0, 0.0, 1.0];
        }
        let violation_after_move = reg.check_solidity_violation(&TokenId::new("cup_1"), &TokenId::new("stone_1"));
        assert!(!violation_after_move);
    }
}
