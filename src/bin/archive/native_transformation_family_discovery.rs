use cdt_rqm_epr::transformation_family_discovery::{
    run_family_discovery_benchmark, FamilyDiscoveryBenchmarkConfig,
};
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_family_discovery_benchmark(FamilyDiscoveryBenchmarkConfig::default());
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report.decision != "family_parameter_mdl_discovery_pass" {
        return Err(io::Error::other(format!(
            "el descubrimiento de familias no pasó: {}",
            report.decision
        ))
        .into());
    }
    Ok(())
}
