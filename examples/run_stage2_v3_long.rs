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
    md.push_str("Módulo: `src/field_autonomy_stage2_v2.rs`.\n");
    md.push_str("Rama: `exp/field-autonomy-next`.\n");
    let tip = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    md.push_str(&format!("Commit tip: `{tip}`.\n"));
    md.push_str(&format!(
        "Seeds: {} (base DEV 0xA300..0xA307; extra={})\n",
        seeds.len(),
        extra
    ));
    md.push_str("Seeds confirmación `0xB300–0xB30F`: **no corridas**.\n");
    md.push_str("Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.\n\n");
    md.push_str("## Hardening en esta corrida (v3.5)\n\n");
    md.push_str("- `HyperparamLock::long()`: dyn_epochs=560 dyn_updates=7 (push absolute cos_dyn).\n");
    md.push_str("- **E21 dual-probe (v3.5)**: SoftScale train/dyn + Relative-only static probe.\n");
    md.push_str("- **E22 (v3.5)**: compose-chain hop-2 Dφ; adaptive lin∩mlp; Relative static endpoints; stronger OOD decoder.\n");
    md.push_str("- **E23 (v3.5)**: 70/30 short/long TF; light h16 residual free-run (no h32/h64 free-run).\n");
    md.push_str("- Dual FeatPath retained: SoftScale E22/E23 dyn; Relative E18/E24/E25/E27 (+ E21/E22 static probe).\n");
    md.push_str("- Static baseline **sin** action + contraste geom-only; dyn extras + multi-familia + E21 dx denso.\n\n");
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
    md.push_str("\nCSV: [`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)\n\n");
    md.push_str("## Lectura honesta (corrida LONG)\n\n");
    md.push_str("- Leakage FIELD_ONLY y contaminación: ver secciones arriba (deben ser 0).\n");
    md.push_str("- Palancas v3.5: E21 SoftScale+Relative probe; E22 compose-chain+adaptive blend+Relative static; E23 70/30+h16 residual.\n");
    md.push_str("- Seeds confirmación `0xB300–0xB30F`: **no corridas** (DEV no locked aún para confirmación).\n");
    fs::write("docs/resultados_etapa_2_v3_long.md", &md).expect("write md");
    println!("wrote docs/resultados_etapa_2_v3_long.{{md,csv}} ({elapsed:.1}s)");
    for (exp, vh) in &hist {
        println!("  {exp}: {vh:?}");
    }
}
