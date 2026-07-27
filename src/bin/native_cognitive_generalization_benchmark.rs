use cdt_rqm_epr::cognitive_generalization_benchmark::{
    run_cognitive_generalization_benchmark, CognitiveGeneralizationConfig,
};
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = run_cognitive_generalization_benchmark(CognitiveGeneralizationConfig::default());
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report.decision != "limited_structural_generalization_pass" {
        return Err(io::Error::other(format!(
            "el protocolo cognitivo no pasó: {}",
            report.decision
        ))
        .into());
    }
    Ok(())
}
