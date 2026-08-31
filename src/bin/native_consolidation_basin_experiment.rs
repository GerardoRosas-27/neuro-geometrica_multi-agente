use cdt_rqm_epr::basin_external_baselines::run_basin_external_baselines;
use cdt_rqm_epr::consolidation_basin_experiment::{
    run_consolidation_basin_experiment, ConsolidationBasinConfig,
};
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConsolidationBasinConfig::default();
    let report = run_consolidation_basin_experiment(config.clone())?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    let table = run_basin_external_baselines(config)?;
    println!("{}", serde_json::to_string_pretty(&table)?);
    if report.decision != "basin_expansion_pass" {
        return Err(io::Error::other(format!(
            "la consolidación no amplió la cuenca bajo el protocolo fijado: {}",
            report.decision
        ))
        .into());
    }
    Ok(())
}
