//! Full Clean-Room v2 development suite (seeds 0xA300–0xA307).
fn main() {
    use cdt_rqm_epr::field_autonomy_stage2_v2::{
        confirmation_deferred_note, rows_to_csv, run_dev_suite, HyperparamLock, DEV_SEEDS,
    };
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::Instant;

    let smoke_only = std::env::var("STAGE2_V2_SMOKE").ok().as_deref() == Some("1");
    println!(
        "Clean-Room v2 {} suite over {} seeds ({})",
        if smoke_only { "SMOKE" } else { "FULL" },
        DEV_SEEDS.len(),
        confirmation_deferred_note()
    );
    let t0 = Instant::now();
    let rows = run_dev_suite(smoke_only);
    let elapsed = t0.elapsed().as_secs_f64();

    let csv = rows_to_csv(&rows);
    fs::write("docs/resultados_etapa_2_cleanroom_v2.csv", &csv).expect("write csv");

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
    md.push_str("# Resultados — Etapa 2 Clean-Room v2\n\n");
    md.push_str(
        "Protocolo: `docs/etapa_2_autonomia_campo_experimentos.md` **§29+** (prevalece).\n",
    );
    md.push_str("Módulo: `src/field_autonomy_stage2_v2.rs` (legacy `field_autonomy_stage2.rs` = histórico / pre-cleanroom).\n");
    md.push_str("Rama: `exp/field-autonomy-next`\n");
    md.push_str(&format!(
        "Seeds desarrollo: 8 × `0xA300..0xA307` ({})\n",
        if smoke_only {
            "smoke subset"
        } else {
            "full suite"
        }
    ));
    md.push_str("Seeds confirmación `0xB300–0xB30F`: **reservadas / diferidas** (no corridas; lock solo sobre DEV).\n");
    md.push_str("Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.\n\n");
    md.push_str("## Nota sobre resultados previos\n\n");
    md.push_str("`docs/resultados_etapa_2_autonomia.md` / `.csv` son **pre-cleanroom / históricos** (seeds `0xE1800..`). No mezclar con este informe.\n\n");
    md.push_str(&format!(
        "Wall time: **{elapsed:.1}s**. Filas: {}.\n\n",
        rows.len()
    ));
    md.push_str("## Hyperparam lock\n\n");
    let mut hp = if smoke_only {
        HyperparamLock::smoke()
    } else {
        HyperparamLock::from_env()
    };
    hp.lock();
    md.push_str(&format!(
        "- Flujo: TRAIN→DEV→LOCK→TEST\n- Status tras suite: **{}**\n- enc_epochs={} dyn_epochs={} dyn_updates={} train_n/dev_n/test_n={}/{}/{}\n- Confirmation no usada → TEST no invalidado por retune post-TEST.\n\n",
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
        "- Filas con `DATASET_INVALID`: **{contam_invalid}**\n- Auditor hard: exact/near/equivalent_state/equivalent_target; same_orbit & equivalent_transformation cubiertos en tests unitarios.\n\n"
    ));
    md.push_str("## Leakage\n\n");
    md.push_str(&format!(
        "- Filas con leakage>0: **{leak_nonzero}** (objetivo FIELD_ONLY = 0)\n\n"
    ));
    md.push_str("## Histogramas de veredicto por experimento\n\n");
    md.push_str("| Experimento | verdict hist |\n|---|---|\n");
    for (exp, vh) in &hist {
        let parts: Vec<String> = vh.iter().map(|(k, v)| format!("{k}:{v}")).collect();
        md.push_str(&format!("| `{exp}` | {} |\n", parts.join(", ")));
    }
    md.push_str("\n## Implementado vs diferido\n\n");
    md.push_str("| Estado | Ítems |\n|---|---|\n");
    md.push_str("| Landed | Generator+auditor+sealed manifests; E18; E18C C0–C6; E19 provenance; E20 antimem; E21 interp/extrap; E22 compose RQM-OFF; E23 rollout h≤64 no TF; E24 paired+bootstrap; E25 cycle; E26 ablation; E27 transfer; E28 continual; E29 intervene; E30 serialize/reload (in-process) |\n");
    md.push_str("| Deferred | Confirmation seeds 0xB300–0xB30F; true OS process restart in E30; Gemma linguistic periphery benchmark |\n\n");
    md.push_str(
        "CSV: [`resultados_etapa_2_cleanroom_v2.csv`](resultados_etapa_2_cleanroom_v2.csv)\n",
    );
    fs::write("docs/resultados_etapa_2_cleanroom_v2.md", &md).expect("write md");
    println!("wrote docs/resultados_etapa_2_cleanroom_v2.{{md,csv}} ({elapsed:.1}s)");
    for (exp, vh) in &hist {
        println!("  {exp}: {vh:?}");
    }
}
