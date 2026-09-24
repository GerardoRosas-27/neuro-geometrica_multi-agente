//! Confirmation seeds 0xB300–0xB30F after DEV majority POS+STRONG on E18/E21/E22/E24.
fn main() {
    use cdt_rqm_epr::field_autonomy_stage2_v2::{
        rows_to_csv, run_dev_suite_with, HyperparamLock, CONFIRMATION_SEEDS,
    };
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::Instant;

    let mut hp = HyperparamLock::long();
    hp.lock();
    let seeds: Vec<u64> = CONFIRMATION_SEEDS.to_vec();
    println!(
        "Clean-Room v3 CONFIRMATION: {} seeds 0xB300–0xB30F, enc={} dyn={}",
        seeds.len(),
        hp.enc_epochs,
        hp.dyn_epochs
    );
    let t0 = Instant::now();
    let rows = run_dev_suite_with(false, &hp, &seeds);
    let elapsed = t0.elapsed().as_secs_f64();
    let csv = rows_to_csv(&rows);
    fs::create_dir_all("docs").ok();
    fs::write("docs/resultados_etapa_2_v3_confirm.csv", &csv).expect("csv");

    let mut hist: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for r in &rows {
        *hist
            .entry(r.experiment.clone())
            .or_default()
            .entry(r.verdict.clone())
            .or_default() += 1;
    }
    let tip = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let mut md = String::new();
    md.push_str("# Confirmación — Etapa 2 Clean-Room v3 (0xB300–0xB30F)\n\n");
    md.push_str(&format!("Rama: `exp/field-autonomy-next`. Commit tip: `{tip}`.\n"));
    md.push_str("Hyperparams: same `HyperparamLock::long()` locked on DEV; no retune.\n");
    md.push_str(&format!("Wall time: **{elapsed:.1}s**. Filas: {}.\n\n", rows.len()));
    md.push_str("## Histogramas\n\n| Experimento | verdict hist |\n|---|---|\n");
    for (exp, vh) in &hist {
        let parts: Vec<String> = vh.iter().map(|(k, v)| format!("{k}:{v}")).collect();
        md.push_str(&format!("| `{exp}` | {} |\n", parts.join(", ")));
    }
    md.push_str("\nCSV: [`resultados_etapa_2_v3_confirm.csv`](resultados_etapa_2_v3_confirm.csv)\n");
    fs::write("docs/resultados_etapa_2_v3_confirm.md", &md).expect("md");
    println!("wrote docs/resultados_etapa_2_v3_confirm.{{md,csv}} ({elapsed:.1}s)");
    for (exp, vh) in &hist {
        if exp.contains("E18_reinforced")
            || exp.contains("E21_")
            || exp.contains("E22_")
            || exp.contains("E24_")
        {
            println!("  {exp}: {vh:?}");
        }
    }
}
