//! Módulo de Conectores Inter-Arquitectónicos.
//!
//! Conecta y reutiliza las arquitecturas construidas en todas las ramas del repositorio:
//! 1. continuous_field_connector: Acoplamiento gemma_campo & campo_turno con Sistema 1.
//! 2. bio_quantum_binding_connector: Acoplamiento BQ10 & ATLAS-FW W^F con ligamiento algebraico.
//! 3. simplicial_spelke_connector: Acoplamiento Soft-Bellman Simplicial 3D & CDT con física Spelke y Theory of Mind.
//! 4. constellation_type_token_connector: Acoplamiento Constelación v4 & ÓRBITA con jerarquía Tipo-Token.
//! 5. unified_marcus_substrate: El sustrato cognitivo unificado que orquesta la arquitectura completa.

pub mod continuous_field_connector;
pub mod bio_quantum_binding_connector;
pub mod simplicial_spelke_connector;
pub mod constellation_type_token_connector;
pub mod unified_marcus_substrate;

pub use continuous_field_connector::{
    ContinuousFieldPerceptualConnector, ContinuousFieldState, SemanticConceptAnchor,
};
pub use bio_quantum_binding_connector::{
    BioQuantumBindingConfig, BioQuantumBindingConnector, BioQuantumFastWeightMatrix,
};
pub use simplicial_spelke_connector::{
    Simplex3D, SimplicialSpelkeSpatialConnector,
};
pub use constellation_type_token_connector::{
    ConstellationNode, ConstellationTypeTokenConnector,
};
pub use unified_marcus_substrate::{
    MarcusUnifiedSubstrate, UnifiedSubstrateTelemetry,
};
