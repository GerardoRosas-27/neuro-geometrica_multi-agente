//! v6 runner (harness design reused from v5).
//!   run_v6 run --phase smoke|dev|confirm [--out artifacts/v6]
//!   run_v6 eval-checkpoint ...   (fresh-process, same contract as v4)
//!
//! Smoke uses SMOKE_STEPS (env V6_SMOKE_STEPS / V6_ARMS allowed only in
//! smoke) and is not evidence. DEV and CONFIRM use the pre-registered budget
//! in `docs/preregistro_v6.md` and refuse any override.

use cdt_rqm_epr::field_autonomy_v4::eval_checkpoint_cli;
use cdt_rqm_epr::v6_hypo::{
    self as v6, CONFIRM_SEEDS_V6, DEV_SEEDS_V6, SMOKE_SEED_V6, SMOKE_STEPS,
};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

/// Commits whose per-seed results are bit-identical to HEAD's (only infra
/// changed after them: checkpointing, job count). Cached seeds from these
/// SHAs are reused. See docs/comparacion_v4_v5_preregistro.md.
const RESULT_COMPAT_SHAS: &[&str] = &[];

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
    let smoke_steps = std::env::var("V6_SMOKE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SMOKE_STEPS);
    let (seeds, steps): (Vec<u64>, usize) = match phase.as_str() {
        "dev" => (DEV_SEEDS_V6.to_vec(), v6::V6_STEPS),
        "confirm" => (CONFIRM_SEEDS_V6.to_vec(), v6::V6_STEPS),
        "smoke" => (vec![SMOKE_SEED_V6], smoke_steps),
        p => panic!("unknown phase {p}"),
    };
    let only: Option<Vec<String>> = std::env::var("V6_ARMS")
        .ok()
        .map(|v| v.split(',').map(String::from).collect());
    if phase != "smoke" && (only.is_some() || std::env::var("V6_SMOKE_STEPS").is_ok()) {
        panic!("V6_ARMS / V6_SMOKE_STEPS are smoke-only (pre-registered DEV/CONFIRM)");
    }
    let out = PathBuf::from(get("--out").unwrap_or_else(|| "artifacts/v6".into())).join(&phase);
    fs::create_dir_all(&out).expect("out dir");
    let sha = git(&["rev-parse", "HEAD"]);
    let dirty = !git(&["status", "--porcelain", "--untracked-files=no"]).is_empty();
    let opts = cdt_rqm_epr::field_autonomy_v4::RunOpts {
        exps: std::collections::HashSet::new(),
        ckpt_dir: Some(out.join("checkpoints")),
        exe: std::env::current_exe().ok(),
    };
    // Infra only: intra-seed checkpoints (V6_CKPT_SECS, default 300 s, or V6_CKPT_STEPS).
    let train_ckpt = out.join("train_ckpt");
    // Infra only: number of seeds trained concurrently (V6_JOBS, default all
    // cores). Each seed is single-threaded and independent; results do not
    // depend on this.
    if let Some(j) = std::env::var("V6_JOBS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(j)
            .build_global()
            .ok();
    }
    let t0 = std::time::Instant::now();
    // Per-seed cache so an interrupted run resumes without recomputing
    // finished seeds (same binary/SHA required; see `sha` in each file).
    let cache = out.join("seeds");
    fs::create_dir_all(&cache).expect("cache dir");
    let results: Vec<(u64, serde_json::Value)> = seeds
        .par_iter()
        .map(|&s| {
            let f = cache.join(format!("0x{s:X}.json"));
            if let Some(v) = fs::read_to_string(&f)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .filter(|v| {
                    v["steps"] == steps
                        && v["arms"] == serde_json::json!(only)
                        && (v["sha"] == sha.as_str()
                            || RESULT_COMPAT_SHAS.contains(&v["sha"].as_str().unwrap_or("")))
                })
            {
                eprintln!("seed 0x{s:X} cached");
                return (s, v);
            }
            let r = v6::run_seed(s, steps, &opts, Some(&train_ckpt), only.as_deref());
            let v = serde_json::json!({
                "sha": sha,
                "steps": steps,
                "arms": only,
                "manifest": r.manifest_json,
                "rows": r.rows.iter().map(|x| x.csv()).collect::<Vec<_>>(),
                "gates": r.rows.iter().filter(|x| x.condition == "GATE")
                    .map(|x| vec![x.partition.clone(), x.status.clone(), x.notes.clone()])
                    .collect::<Vec<_>>(),
            });
            fs::write(&f, v.to_string()).ok();
            {
                for e in fs::read_dir(&train_ckpt).into_iter().flatten().flatten() {
                    if e.file_name()
                        .to_string_lossy()
                        .starts_with(&format!("0x{s:X}_"))
                    {
                        fs::remove_file(e.path()).ok();
                    }
                }
            }
            eprintln!("seed 0x{s:X} done ({:.0}s)", t0.elapsed().as_secs_f64());
            (s, v)
        })
        .collect();
    let mut csv = String::from(v6::csv_header());
    csv.push('\n');
    let mut manifests = String::new();
    let mut gates: BTreeMap<String, Vec<(u64, String, String)>> = BTreeMap::new();
    for (s, r) in &results {
        for row in r["rows"].as_array().into_iter().flatten() {
            csv.push_str(row.as_str().unwrap_or(""));
            csv.push('\n');
        }
        for g in r["gates"].as_array().into_iter().flatten() {
            let f = |i: usize| g[i].as_str().unwrap_or("").to_string();
            gates.entry(f(0)).or_default().push((*s, f(1), f(2)));
        }
        let _ = writeln!(manifests, "{}", r["manifest"].as_str().unwrap_or(""));
    }
    fs::write(out.join("metrics.csv"), &csv).unwrap();
    fs::write(out.join("dataset_manifest.jsonl"), &manifests).unwrap();
    let config = format!(
        "{{\"git_sha\":\"{sha}\",\"git_dirty\":{dirty},\"branch\":\"exp/v6-hipotesis\",\"phase\":\"{phase}\",\"steps\":{steps},\"seeds\":[{}],\"wall_s\":{:.1}}}",
        seeds
            .iter()
            .map(|s| format!("\"0x{s:X}\""))
            .collect::<Vec<_>>()
            .join(","),
        t0.elapsed().as_secs_f64()
    );
    fs::write(out.join("config.json"), &config).unwrap();
    let mut md = format!(
        "# v6 {phase} summary\n\n- git SHA: `{sha}` (dirty={dirty})\n- steps: {steps}\n- seeds: {}\n\n| Gate | PASS | FAIL | n |\n|---|---|---|---|\n",
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
