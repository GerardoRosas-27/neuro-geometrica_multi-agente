use cdt_rqm_epr::consolidation_basin_experiment::{
    run_consolidation_basin_experiment, ConsolidationBasinConfig,
};
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_consolidation_basin_experiment(ConsolidationBasinConfig::default())?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report.decision != "basin_expansion_pass" {
        return Err(io::Error::other(format!(
            "la consolidación no amplió la cuenca bajo el protocolo fijado: {}",
            report.decision
        ))
        .into());
    }
    Ok(())
}
