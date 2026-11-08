//! Módulo Algebraic Mind.
//!
//! Reúne las cuatro capacidades clave de "The Algebraic Mind" (Gary Marcus, 2001):
//! 1. Operaciones sobre variables y ligamiento dinámico (variables.rs).
//! 2. Sintaxis jerárquica y composicionalidad recursiva (syntax_tree.rs).
//! 3. Distinción ontológica Tipo vs Token (type_token.rs).
//! 4. Generalización sistemática fuera de distribución (OOD).

pub mod variables;
pub mod syntax_tree;
pub mod type_token;

pub use variables::{
    AlgebraicPattern, AlgebraicRuleEngine, HrrVector, VarId, VarValue, VariableRegister,
};
pub use syntax_tree::SyntaxNode;
pub use type_token::{AbstractType, TokenId, TokenInstance, TypeId, TypeTokenRegistry};
