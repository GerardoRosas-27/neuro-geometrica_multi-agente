use cdt_rqm_epr::advanced_cognitive_validation::{
    run_advanced_cognitive_validation, AdvancedCognitiveValidationConfig,
};
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_advanced_cognitive_validation(AdvancedCognitiveValidationConfig::default());
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report.decision != "adversarial_selection_and_limited_symmetry_discovery_pass" {
        return Err(io::Error::other(format!(
            "la validación cognitiva adversarial no pasó: {}",
            report.decision
        ))
        .into());
    }
    Ok(())
}
