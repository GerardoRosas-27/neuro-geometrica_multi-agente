use cdt_rqm_epr::basin_external_baselines::{
    run_basin_external_baselines, run_capacity_curve, CapacityCurveConfig,
};
use cdt_rqm_epr::consolidation_basin_experiment::{
    run_bounded_forgetting, run_consolidation_basin_experiment, BoundedForgettingConfig,
    ConsolidationBasinConfig,
};
use serde::Serialize;
use std::io;

#[derive(Serialize)]
struct CanonicalExperiment {
    basin: cdt_rqm_epr::consolidation_basin_experiment::ConsolidationBasinReport,
    baselines: cdt_rqm_epr::basin_external_baselines::BasinBaselineTable,
    bounded_forgetting: cdt_rqm_epr::consolidation_basin_experiment::BoundedForgettingReport,
    capacity: cdt_rqm_epr::basin_external_baselines::CapacityCurveReport,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConsolidationBasinConfig::default();
    let basin = run_consolidation_basin_experiment(config.clone())?;
    let baselines = run_basin_external_baselines(config)?;
    let bounded_forgetting = run_bounded_forgetting(BoundedForgettingConfig::default())?;
    let capacity = run_capacity_curve(CapacityCurveConfig::default())?;
    let report = CanonicalExperiment {
        basin,
        baselines,
        bounded_forgetting,
        capacity,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report.basin.decision != "basin_expansion_pass" {
        return Err(io::Error::other(format!(
            "la consolidación no amplió la cuenca bajo el protocolo fijado: {}",
            report.basin.decision
        ))
        .into());
    }
    Ok(())
}
