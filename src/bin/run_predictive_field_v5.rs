//! v5 runner.
//!   run_predictive_field_v5 run --phase smoke|dev|confirm [--out artifacts/v5]
//!   run_predictive_field_v5 eval-checkpoint ...   (fresh-process, same contract as v4)
//!
//! Smoke uses 200 steps and is not evidence. DEV and CONFIRM use the
//! pre-registered budget in `docs/preregistro_v5.md`.

use cdt_rqm_epr::field_autonomy_v4::{
    eval_checkpoint_cli, CONFIRM_SEEDS_V4, DEV_SEEDS_V4, SMOKE_SEED_V4,
};
use cdt_rqm_epr::v5_predict::{self, SMOKE_STEPS};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("eval-checkpoint") {
        std::process::exit(eval_checkpoint_cli(&args[2..]));
    }
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let phase = get("--phase").unwrap_or_else(|| "smoke".into());
    let (seeds, steps): (Vec<u64>, usize) = match phase.as_str() {
        "dev" => (DEV_SEEDS_V4.to_vec(), v5_predict::full_steps()),
        "confirm" => (CONFIRM_SEEDS_V4.to_vec(), v5_predict::full_steps()),
        _ => (vec![SMOKE_SEED_V4], SMOKE_STEPS),
    };
    let out = PathBuf::from(get("--out").unwrap_or_else(|| "artifacts/v5".into())).join(&phase);
    fs::create_dir_all(&out).expect("out dir");
    let sha = git(&["rev-parse", "HEAD"]);
    let dirty = !git(&["status", "--porcelain", "--untracked-files=no"]).is_empty();
    let opts = cdt_rqm_epr::field_autonomy_v4::RunOpts {
        exps: std::collections::HashSet::new(),
        ckpt_dir: Some(out.join("checkpoints")),
        exe: std::env::current_exe().ok(),
    };
    let t0 = std::time::Instant::now();
    let results: Vec<(u64, v5_predict::SeedOut)> = seeds
        .par_iter()
        .map(|&s| {
            let r = v5_predict::run_seed(s, steps, &opts);
            eprintln!("seed 0x{s:X} done ({:.0}s)", t0.elapsed().as_secs_f64());
            (s, r)
        })
        .collect();
    let mut csv = String::from(v5_predict::csv_header());
    csv.push('\n');
    let mut manifests = String::new();
    let mut gates: BTreeMap<String, Vec<(u64, String, String)>> = BTreeMap::new();
    for (s, r) in &results {
        for row in &r.rows {
            csv.push_str(&row.csv());
            csv.push('\n');
            if row.condition == "GATE" {
                gates.entry(row.partition.clone()).or_default().push((
                    *s,
                    row.status.clone(),
                    row.notes.clone(),
                ));
            }
        }
        let _ = writeln!(manifests, "{}", r.manifest_json);
    }
    fs::write(out.join("metrics.csv"), &csv).unwrap();
    fs::write(out.join("dataset_manifest.jsonl"), &manifests).unwrap();
    let config = format!(
        "{{\"git_sha\":\"{sha}\",\"git_dirty\":{dirty},\"branch\":\"exp/predictive-field-v5\",\"phase\":\"{phase}\",\"steps\":{steps},\"seeds\":[{}],\"wall_s\":{:.1}}}",
        seeds
            .iter()
            .map(|s| format!("\"0x{s:X}\""))
            .collect::<Vec<_>>()
            .join(","),
        t0.elapsed().as_secs_f64()
    );
    fs::write(out.join("config.json"), &config).unwrap();
    let mut md = format!(
        "# v5 {phase} summary\n\n- git SHA: `{sha}` (dirty={dirty})\n- steps: {steps}\n- seeds: {}\n\n| Gate | PASS | FAIL | n |\n|---|---|---|---|\n",
        seeds.len()
    );
    for (e, v) in &gates {
        let p = v.iter().filter(|x| x.1 == "PASS").count();
        let _ = writeln!(md, "| {e} | {p} | {} | {} |", v.len() - p, v.len());
    }
    md.push_str("\n## Per-seed gate notes\n\n");
    for (e, v) in &gates {
        let _ = writeln!(md, "### {e}\n");
        for (s, st, n) in v {
            let _ = writeln!(md, "- 0x{s:X} **{st}** — {n}");
        }
        md.push('\n');
    }
    fs::write(out.join("summary.md"), &md).unwrap();
    println!("{md}");
}
