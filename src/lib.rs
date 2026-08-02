pub mod adaptive_gemma2;
pub mod advanced_cognitive_validation;
pub mod cognitive_generalization_benchmark;
pub mod cognitive_logistics;
pub mod cognitive_os;
pub mod consolidation_basin_experiment;
/// Gate de cognición emergente. Estado: sólo tests propios; el entrenador
/// infinito reimplementa su propio gate. Pendiente decidir integración real o
/// retiro (ver análisis de módulos huérfanos, jul-2026).
pub mod emergent_cognition_training;
/// Comparador de motores. Estado: sólo tests propios, sin binario asociado.
pub mod engine_comparison;
pub mod entanglement;
pub mod gemma_operator_bridge;
pub mod gemma_phasor_coupling;
pub mod matrix_free_cognitive_substrate;
pub mod native_checkpoint;
pub mod native_cognitive_closed_loop;
pub mod native_gemma2;
pub mod native_gemma2_runtime;
pub mod native_hybrid_phasor_cdt_engine;
pub mod native_multi_operator_core;
pub mod native_phasor_thermodynamic_engine;
/// Red plástica de simetría. Estado: sólo tests propios; ningún módulo ni
/// binario la consume todavía.
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
pub mod transformation_family_discovery;
pub mod unified_spin_cognitive_engine;
/// VMC Jastrow complejo. Estado: validado contra energía exacta en tests,
/// aún sin consumidores fuera de ellos; candidato a backend variacional del
/// motor unificado.
pub mod variational_spin_liquid_vmc;
