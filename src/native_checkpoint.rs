//! Persistencia reutilizable y transaccional del sustrato nativo.

use crate::native_thermo_rqm_epr::NativeThermoRqmEprSubstrate;
use std::fs;
use std::path::{Path, PathBuf};

pub fn serialize_native_checkpoint(substrate: &NativeThermoRqmEprSubstrate) -> String {
    let mut out = String::from("NATIVE_THERMO_RQM_EPR_CLEAN_STATE_V1\n");
    out.push_str("stats 0 0 0 0 0 0 1 0\n");
    let cdt = substrate.thermal.config;
    out.push_str(&format!(
        "thermal_config {} {} {} {} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {}\n",
        cdt.slices,
        cdt.nodes_per_slice,
        cdt.spatial_degree,
        cdt.temporal_degree,
        cdt.temperature,
        cdt.dt,
        cdt.diffusion,
        cdt.confinement,
        cdt.pilot_gain,
        cdt.phase_coupling,
        cdt.amplitude_decay,
        cdt.state_clamp,
        cdt.seed
    ));
    let rqm = substrate.config;
    out.push_str(&format!(
        "rqm_config {:.7} {:.7} {:.7} {:.7} {:.7} {} {} {:.7} {:.7} {} {} {} {} {} {}\n",
        rqm.amplitude_learning_rate,
        rqm.coherence_learning_rate,
        rqm.uncertainty_learning_rate,
        rqm.phase_learning_rate,
        rqm.amplitude_decay,
        rqm.thermal_steps_per_train,
        rqm.thermal_steps_per_query,
        rqm.thermal_score_gain,
        rqm.thermal_activation_margin,
        usize::from(rqm.collect_query_diagnostics),
        rqm.max_candidates,
        rqm.max_pilot_window_nodes,
        rqm.sampling_block_size,
        rqm.sampling_schedule_rounds,
        rqm.max_sampling_blocks
    ));
    out.push_str(&format!("nodes {}\n", substrate.thermal.node_count()));
    for node in 0..substrate.thermal.node_count() {
        out.push_str(&format!(
            "n {} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7}\n",
            node,
            substrate.thermal.thermal_state[node],
            substrate.thermal.amplitude[node],
            substrate.thermal.phase[node],
            substrate.thermal.temperature[node],
            substrate.thermal.energy[node],
            substrate.thermal.activation[node]
        ));
    }
    let relations = substrate.relation_entries().collect::<Vec<_>>();
    out.push_str(&format!("relations {}\n", relations.len()));
    for (observer, source, target, amplitude, phase, coherence, uncertainty, last_tick) in relations
    {
        out.push_str(&format!(
            "r {} {} {} {:.7} {:.7} {:.7} {:.7} {}\n",
            observer.0, source, target, amplitude, phase, coherence, uncertainty, last_tick
        ));
    }
    out.push_str("entanglement_begin\n");
    out.push_str(&substrate.entanglement.serialize_persistent_state());
    out.push_str("entanglement_end\nend\n");
    out
}

pub fn save_native_checkpoint_transactional(
    substrate: &NativeThermoRqmEprSubstrate,
    destination: impl AsRef<Path>,
) -> Result<(), String> {
    atomic_write(
        destination.as_ref(),
        serialize_native_checkpoint(substrate).as_bytes(),
    )
}

pub fn atomic_write(destination: &Path, body: &[u8]) -> Result<(), String> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temporary = with_suffix(destination, "tmp");
    let backup = with_suffix(destination, "bak");
    fs::write(&temporary, body).map_err(|error| error.to_string())?;
    if backup.exists() {
        fs::remove_file(&backup).map_err(|error| error.to_string())?;
    }
    if destination.exists() {
        fs::rename(destination, &backup).map_err(|error| error.to_string())?;
    }
    match fs::rename(&temporary, destination) {
        Ok(()) => {
            if backup.exists() {
                fs::remove_file(backup).map_err(|error| error.to_string())?;
            }
            Ok(())
        }
        Err(error) => {
            if backup.exists() {
                let _ = fs::rename(&backup, destination);
            }
            Err(error.to_string())
        }
    }
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(format!(".{suffix}"));
    PathBuf::from(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entanglement::EntanglementConfig;
    use crate::native_thermo_rqm_epr::NativeThermoRqmConfig;
    use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;

    #[test]
    fn serialized_checkpoint_uses_loadable_native_header() {
        let substrate = NativeThermoRqmEprSubstrate::new(
            NativeThermoCdtConfig {
                slices: 2,
                nodes_per_slice: 8,
                ..NativeThermoCdtConfig::default()
            },
            NativeThermoRqmConfig::default(),
            EntanglementConfig::default(),
        );
        let serialized = serialize_native_checkpoint(&substrate);
        assert!(serialized.starts_with("NATIVE_THERMO_RQM_EPR_CLEAN_STATE_V1\n"));
        assert!(serialized.contains("\nentanglement_begin\n"));
        assert!(serialized.ends_with("entanglement_end\nend\n"));
    }
}
