//! Long / varied Clean-Room suite for plan v3 on `exp/field-autonomy-next`.
//! Uses HyperparamLock::long(); optional STAGE2_V3_EXTRA_SEEDS=1 adds 0xA308–0xA30F.
fn main() {
    use cdt_rqm_epr::field_autonomy_stage2_v2::{
        confirmation_deferred_note, rows_to_csv, run_dev_suite_with, HyperparamLock, DEV_SEEDS,
    };
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::Instant;

    let mut hp = HyperparamLock::long();
    hp.lock();
    let mut seeds: Vec<u64> = DEV_SEEDS.to_vec();
    let extra = std::env::var("STAGE2_V3_EXTRA_SEEDS").ok().as_deref() == Some("1");
    if extra {
        seeds.extend([
            0xA308u64, 0xA309, 0xA30A, 0xA30B, 0xA30C, 0xA30D, 0xA30E, 0xA30F,
        ]);
    }
    println!(
        "Clean-Room v3 LONG suite: {} seeds, enc={} dyn={} updates={} train_n={} ({})",
        seeds.len(),
        hp.enc_epochs,
        hp.dyn_epochs,
        hp.dyn_updates,
        hp.train_n,
        confirmation_deferred_note()
    );
    let t0 = Instant::now();
    let rows = run_dev_suite_with(false, &hp, &seeds);
    let elapsed = t0.elapsed().as_secs_f64();

    let csv = rows_to_csv(&rows);
    fs::create_dir_all("docs").ok();
    fs::write("docs/resultados_etapa_2_v3_long.csv", &csv).expect("write csv");

    let mut hist: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut leak_nonzero = 0usize;
    let mut contam_invalid = 0usize;
    for r in &rows {
        *hist
            .entry(r.experiment.clone())
            .or_default()
            .entry(r.verdict.clone())
            .or_default() += 1;
        if r.leakage_score > 0 {
            leak_nonzero += 1;
        }
        if r.contamination == "DATASET_INVALID" {
            contam_invalid += 1;
        }
    }

    let mut md = String::new();
    md.push_str("# Resultados — Etapa 2 Clean-Room v3 LONG\n\n");
    md.push_str("Protocolo: `docs/plan_autonomia_campo_v3.md` + `docs/etapa_2_autonomia_campo_experimentos.md` §29+.\n");
    md.push_str("Módulo: `src/field_autonomy_stage2_v2.rs`\n");
    md.push_str("Rama: `exp/field-autonomy-next`\n");
    md.push_str(&format!(
        "Seeds: {} (base DEV 0xA300..0xA307; extra={})\n",
        seeds.len(),
        extra
    ));
    md.push_str("Seeds confirmación `0xB300–0xB30F`: **no corridas**.\n");
    md.push_str("Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.\n\n");
    md.push_str("## Hardening en esta corrida\n\n");
    md.push_str("- `HyperparamLock::long()`: más datos/épocas/updates.\n");
    md.push_str("- Action cues para translation/rotation/scaling/affine/compose.\n");
    md.push_str("- E21: curriculum dx más denso + augment.\n");
    md.push_str("- E22: eval secuencial oracle-mid (T1 luego T2) RQM-OFF; single-shot en notes.\n\n");
    md.push_str(&format!(
        "Wall time: **{elapsed:.1}s**. Filas: {}.\n\n",
        rows.len()
    ));
    md.push_str("## Hyperparam lock\n\n");
    md.push_str(&format!(
        "- Status: **{}**\n- enc_epochs={} dyn_epochs={} dyn_updates={} train_n/dev_n/test_n={}/{}/{}\n\n",
        hp.status(),
        hp.enc_epochs,
        hp.dyn_epochs,
        hp.dyn_updates,
        hp.train_n,
        hp.dev_n,
        hp.test_n
    ));
    md.push_str("## Contaminación\n\n");
    md.push_str(&format!(
        "- Filas `DATASET_INVALID`: **{contam_invalid}**\n\n"
    ));
    md.push_str("## Leakage\n\n");
    md.push_str(&format!(
        "- Filas leakage>0: **{leak_nonzero}**\n\n"
    ));
    md.push_str("## Histogramas de veredicto\n\n");
    md.push_str("| Experimento | verdict hist |\n|---|---|\n");
    for (exp, vh) in &hist {
        let parts: Vec<String> = vh.iter().map(|(k, v)| format!("{k}:{v}")).collect();
        md.push_str(&format!("| `{exp}` | {} |\n", parts.join(", ")));
    }
    md.push_str("\nCSV: [`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)\n");
    fs::write("docs/resultados_etapa_2_v3_long.md", &md).expect("write md");
    println!("wrote docs/resultados_etapa_2_v3_long.{{md,csv}} ({elapsed:.1}s)");
    for (exp, vh) in &hist {
        println!("  {exp}: {vh:?}");
    }
}
