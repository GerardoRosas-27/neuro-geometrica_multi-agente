pub mod adaptive_gemma2;
pub mod advanced_cognitive_validation;
pub mod basin_external_baselines;
pub mod cognitive_generalization_benchmark;
pub mod cognitive_logistics;
pub mod cognitive_os;
pub mod consolidation_basin_experiment;
/// Gate de cognición emergente. El entrenador infinito no lo usa; sale del
/// crate público (feature `research`). Ver docs/archive.md.
#[cfg(feature = "research")]
pub mod emergent_cognition_training;
/// Comparador huérfano. Feature `research`. Ver docs/archive.md.
#[cfg(feature = "research")]
pub mod engine_comparison;
pub mod entanglement;
/// Entrenamiento donde un prior generativo propone futuros y F postselecciona
/// qué trayectoria puede pasar al gate de consolidación CDT.
pub mod future_guided_training;
pub mod gemma2_circadian_bridge;
pub mod gemma2_thermo_hybrid_llm;
pub mod gemma2_thermo_hybrid_session;
pub mod gemma_future_generator;
pub mod gemma_operator_bridge;
pub mod gemma_phasor_coupling;
pub mod hybrid_thermo_attention;
pub mod hybrid_thermo_attention_comparison;
pub mod matrix_free_cognitive_substrate;
pub mod native_checkpoint;
pub mod native_cognitive_closed_loop;
pub mod native_gemma2;
pub mod native_gemma2_runtime;
pub mod native_hybrid_phasor_cdt_engine;
pub mod native_multi_operator_core;
pub mod native_phasor_thermodynamic_engine;
/// Red plástica de simetría. Nadie la consume. Feature `research`.
/// Ver docs/archive.md.
#[cfg(feature = "research")]
pub mod native_plastic_symmetry_network;
pub mod native_rng;
pub mod native_thermo_rqm_epr;
pub mod native_thermodynamic_cdt;
pub mod native_thermodynamic_engine;
pub mod oxicuda_peps3d_backend;
pub mod oxicuda_pyrochlore_backend;
pub mod plasticity_controller;
pub mod pyrochlore_graph_tensor_network;
pub mod quantum_spin_thermodynamic_engine;
pub mod relational_field;
pub mod residue_budget;
pub mod residue_vacuum_bridge;
pub mod residue_vacuum_fluctuation;
pub mod simplicial_thermodynamic_engine;
pub mod symmetry_guided_rqm_epr;
pub mod symmetry_thermodynamic_substrate;
pub mod thermo_router;
pub mod thermodynamic_attractor_comparison;
/// Comparación de estructuras temporales de inferencia: evidencia sola,
/// evidencia como frontera y dos vectores de estado con post-selección.
pub mod transactional_training_experiment;
pub mod transformation_family_discovery;
pub mod unified_spin_cognitive_engine;
/// VMC Jastrow complejo. El motor unificado puede invocarlo; no alimenta
/// la tesis de cuenca. Ver docs/archive.md.
pub mod variational_spin_liquid_vmc;
