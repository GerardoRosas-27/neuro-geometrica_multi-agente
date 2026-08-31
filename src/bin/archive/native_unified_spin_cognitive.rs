//! Entrada única del motor CDT–spin–RQM–EPR–cognición.

use cdt_rqm_epr::matrix_free_cognitive_substrate::LatentConceptId;
use cdt_rqm_epr::oxicuda_pyrochlore_backend::PyrochloreMpoConfig;
use cdt_rqm_epr::relational_field::ObserverId;
use cdt_rqm_epr::unified_spin_cognitive_engine::{
    UnifiedSpinCognitiveConfig, UnifiedSpinCognitiveEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = UnifiedSpinCognitiveEngine::periodic_pyrochlore(
        2,
        1,
        1,
        UnifiedSpinCognitiveConfig::default(),
    )?;
    let observer = ObserverId(995_001);
    engine.train_relation(
        observer,
        LatentConceptId(0),
        LatentConceptId(1),
        0.25,
        1.0,
        0.0,
        &[],
        40,
    );
    if std::env::args().any(|argument| argument == "--tensor-network") {
        let tensor = engine.refresh_tensor_network(PyrochloreMpoConfig::default())?;
        println!(
            "tensor_network energy={:.9} mpo_bond={}->{} mps_bond={} entropy={:.9}",
            tensor.energy,
            tensor.initial_mpo_bond_dimension,
            tensor.mpo_bond_dimension,
            tensor.max_mps_bond_dimension,
            tensor.midpoint_entropy
        );
    }
    if std::env::args().any(|argument| argument == "--peps3d") {
        let peps = engine.refresh_peps3d_product_scaffold(2, 1, 1, 8)?;
        println!(
            "peps3d grid={}x{}x{} norm={:.9} entropy_z={:.9} local_bonds={} nonlocal_bonds={} optimized={}",
            peps.lx,
            peps.ly,
            peps.lz,
            peps.approximate_norm,
            peps.z_entanglement_entropy,
            peps.local_bonds,
            peps.nonlocal_bonds,
            peps.optimization_available
        );
    }
    if std::env::args().any(|argument| argument == "--graph-tn") {
        let graph = engine.refresh_graph_tensor_network()?;
        println!(
            "graph_tensor sites={} bonds={} tensors={} contraction_cost={} norm={:.9} entropy={:.9} energy={:.9} optimized={}",
            graph.sites,
            graph.represented_bonds,
            graph.tensor_count,
            graph.contraction_cost,
            graph.norm,
            graph.single_spin_entropy,
            graph.heisenberg_energy,
            graph.optimization_available
        );
    }
    engine.train_relation(
        observer,
        LatentConceptId(1),
        LatentConceptId(2),
        0.25,
        1.0,
        0.0,
        &[],
        40,
    );
    let inference = engine
        .infer(observer, LatentConceptId(0), 0.25, 2)
        .ok_or("el motor no produjo inferencia consolidada")?;
    let report = engine.report();
    println!(
        "engine=native_unified_spin_cognitive backend={:?} spins={} tetrahedra={} physical_bonds={}",
        report.backend, report.spins, report.tetrahedra, report.physical_bonds
    );
    println!(
        "cdt_symmetry={:.9} quantum_entropy={:.9} entangled_edges={} rqm_relations={} epr_links={} knowledge={}",
        report.topological_symmetry,
        report.quantum.mean_single_spin_entropy,
        report.quantum.entangled_edges,
        report.rqm_relations,
        report.epr_links,
        report.consolidated_knowledge
    );
    if let Some(energy) = report.tensor_network_energy {
        println!(
            "tensor_network_attached=true energy={:.9} mpo_bond={}",
            energy,
            report.tensor_network_mpo_bond_dimension.unwrap_or(0)
        );
    }
    if let Some(norm) = report.peps3d_norm {
        println!(
            "peps3d_attached=true norm={:.9} entropy={:.9} nonlocal_bonds={}",
            norm,
            report.peps3d_entropy.unwrap_or(0.0),
            report.peps3d_nonlocal_bonds.unwrap_or(0)
        );
    }
    if let Some(bonds) = report.graph_tensor_bonds {
        println!(
            "graph_tensor_attached=true bonds={} entropy={:.9} contraction_cost={}",
            bonds,
            report.graph_tensor_entropy.unwrap_or(0.0),
            report.graph_tensor_contraction_cost.unwrap_or(0)
        );
    }
    println!(
        "cognition path={:?} confidence={:.9} depth={}",
        inference.path, inference.confidence, inference.abstraction_depth
    );
    println!("decision=UNIFIED_ENGINE_READY");
    Ok(())
}
