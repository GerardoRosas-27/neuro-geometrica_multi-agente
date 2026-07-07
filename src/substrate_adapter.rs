use crate::cdt_graphity::CdtGraphityEdgeKind;
use crate::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use crate::entanglement::EntanglementConfig;
use crate::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use crate::native_thermodynamic_cdt::{NativeCdtEdgeKind, NativeThermoCdtConfig};
use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeMigrationSummary {
    pub legacy_relations: usize,
    pub imported_relations: usize,
    pub nodes: usize,
    pub imported_edges: usize,
    pub epr_links: usize,
}

pub fn load_legacy_and_migrate_to_native<P: AsRef<Path>>(
    path: P,
    legacy_config: CdtRqmConfig,
    native_rqm_config: NativeThermoRqmConfig,
    entanglement_config: EntanglementConfig,
) -> io::Result<(
    CdtRqmUniverseSubstrate,
    NativeThermoRqmEprSubstrate,
    NativeMigrationSummary,
)> {
    let mut legacy = CdtRqmUniverseSubstrate::new(legacy_config);
    legacy.load_consolidated_state(path)?;
    let (native, summary) =
        migrate_legacy_to_native(&legacy, native_rqm_config, entanglement_config);
    Ok((legacy, native, summary))
}

pub fn migrate_legacy_to_native(
    legacy: &CdtRqmUniverseSubstrate,
    native_rqm_config: NativeThermoRqmConfig,
    entanglement_config: EntanglementConfig,
) -> (NativeThermoRqmEprSubstrate, NativeMigrationSummary) {
    let mut native = NativeThermoRqmEprSubstrate::new(
        native_thermal_config(legacy),
        native_rqm_config,
        entanglement_config,
    );

    seed_thermal_state(legacy, &mut native);
    seed_geometry(legacy, &mut native);
    seed_relations(legacy, &mut native);
    if let Some(entanglement) = &legacy.entanglement {
        native.entanglement = entanglement.clone();
    }

    let summary = NativeMigrationSummary {
        legacy_relations: legacy.relation_count(),
        imported_relations: native.relation_count(),
        nodes: native.thermal.node_count(),
        imported_edges: native.thermal.edge_count(),
        epr_links: native.entanglement.summary().active_links,
    };
    (native, summary)
}

fn native_thermal_config(legacy: &CdtRqmUniverseSubstrate) -> NativeThermoCdtConfig {
    let cdt = legacy.hardware.config;
    NativeThermoCdtConfig {
        slices: cdt.slices,
        nodes_per_slice: cdt.nodes_per_slice,
        spatial_degree: cdt.target_spatial_degree.max(1),
        temporal_degree: cdt.target_temporal_degree.max(1),
        temperature: legacy.hardware.temperature.max(0.05),
        seed: cdt.seed ^ 0x71E9_4D3A,
        ..NativeThermoCdtConfig::default()
    }
}

fn seed_thermal_state(legacy: &CdtRqmUniverseSubstrate, native: &mut NativeThermoRqmEprSubstrate) {
    for node in &legacy.hardware.nodes {
        if node.id >= native.thermal.node_count() {
            continue;
        }
        native.thermal.thermal_state[node.id] = node.surprise.clamp(-1.0, 1.0);
        native.thermal.amplitude[node.id] =
            (0.5 + node.surprise.abs()).clamp(0.0, native.thermal.config.state_clamp);
        native.thermal.activation[node.id] = if node.activation { 1.0 } else { 0.0 };
        native.thermal.temperature[node.id] = legacy.hardware.temperature.max(0.05);
    }
}

fn seed_geometry(legacy: &CdtRqmUniverseSubstrate, native: &mut NativeThermoRqmEprSubstrate) {
    let edges = legacy
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .map(|edge| {
            let kind = match edge.kind {
                CdtGraphityEdgeKind::Spatial => NativeCdtEdgeKind::Spatial,
                CdtGraphityEdgeKind::Temporal => NativeCdtEdgeKind::Temporal,
            };
            let phase = match kind {
                NativeCdtEdgeKind::Spatial => 0.0,
                NativeCdtEdgeKind::Temporal => 0.25,
            };
            (
                edge.a,
                edge.b,
                kind,
                edge.stability.max(0.01),
                phase,
                edge.stability,
            )
        });
    native.thermal.replace_edges(edges);
}

fn seed_relations(legacy: &CdtRqmUniverseSubstrate, native: &mut NativeThermoRqmEprSubstrate) {
    for (key, state) in legacy.software.relation_entries() {
        native.import_relation_state(
            key.observer,
            key.a,
            key.b,
            state.amplitude,
            state.phase,
            state.coherence,
            state.uncertainty,
            state.last_observed_tick,
        );
        native.import_relation_state(
            key.observer,
            key.b,
            key.a,
            state.amplitude,
            -state.phase,
            state.coherence,
            state.uncertainty,
            state.last_observed_tick,
        );
    }
}
