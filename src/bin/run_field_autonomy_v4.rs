//! v4 runner. Usage:
//!   run_field_autonomy_v4 run --phase smoke|dev|confirm [--exps E31,E35,...] [--out artifacts/v4]
//!   run_field_autonomy_v4 eval-checkpoint ...   (fresh-process evaluator, E37/E38)
use cdt_rqm_epr::field_autonomy_v4::*;
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
    let seeds: Vec<u64> = match phase.as_str() {
        "dev" => DEV_SEEDS_V4.to_vec(),
        "confirm" => CONFIRM_SEEDS_V4.to_vec(),
        _ => vec![SMOKE_SEED_V4],
    };
    let exps: std::collections::HashSet<String> = get("--exps")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();
    if get("--consol").as_deref() == Some("generic") {
        GENERIC_CONSOLIDATION.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let out = PathBuf::from(get("--out").unwrap_or_else(|| "artifacts/v4".into())).join(&phase);
    fs::create_dir_all(&out).expect("out dir");
    let sha = git(&["rev-parse", "HEAD"]);
    let dirty = !git(&["status", "--porcelain", "--untracked-files=no"]).is_empty();
    let opts = RunOpts {
        exps: exps.clone(),
        ckpt_dir: Some(out.join("checkpoints")),
        exe: std::env::current_exe().ok(),
    };
    let t0 = std::time::Instant::now();
    let results: Vec<(u64, SeedOut)> = seeds
        .par_iter()
        .map(|&s| {
            let r = run_seed(s, &opts);
            eprintln!("seed 0x{s:X} done ({:.0}s)", t0.elapsed().as_secs_f64());
            (s, r)
        })
        .collect();
    let mut csv = String::from(CSV_HEADER);
    csv.push('\n');
    let mut prov = String::new();
    let mut manifests = String::new();
    let mut trace = String::from("seed,step,norm_ratio,norm_delta,cosine_target,energy_ratio,manifold_distance,transition_error,jacobian_sigma_max,perturbation_growth_eps[0.001;0.01;0.05;0.10;0.20]@h1/8/32/64\n");
    let mut gates: BTreeMap<String, Vec<(u64, String, String)>> = BTreeMap::new();
    for (s, r) in &results {
        for row in &r.rows {
            csv.push_str(&row.csv());
            csv.push('\n');
            if row.condition == "GATE" {
                gates.entry(row.experiment.clone()).or_default().push((
                    *s,
                    row.status.clone(),
                    row.notes.clone(),
                ));
            }
        }
        for p in &r.provenance {
            prov.push_str(p);
            prov.push('\n');
        }
        let _ = writeln!(manifests, "{}", r.manifest_json);
        for t in &r.e32_trace {
            let _ = writeln!(trace, "{t}");
        }
    }
    fs::write(out.join("metrics.csv"), &csv).unwrap();
    fs::write(out.join("provenance.jsonl"), &prov).unwrap();
    fs::write(out.join("dataset_manifest.jsonl"), &manifests).unwrap();
    fs::write(out.join("e32_trace.csv"), &trace).unwrap();
    let config = format!(
        "{{\"git_sha\":\"{sha}\",\"git_dirty\":{dirty},\"branch\":\"exp/field-autonomy-v4\",\"phase\":\"{phase}\",\"seeds\":[{}],\"hp_lock\":{},\"hp_lock_sha256\":\"{}\",\"exps\":\"{}\",\"consolidation\":\"{}\",\"wall_s\":{:.1}}}",
        seeds.iter().map(|s| format!("\"0x{s:X}\"")).collect::<Vec<_>>().join(","),
        hp_lock_json(),
        hp_lock_hash(),
        get("--exps").unwrap_or_else(|| "ALL".into()),
        get("--consol").unwrap_or_else(|| "rule".into()),
        t0.elapsed().as_secs_f64()
    );
    fs::write(out.join("config.json"), &config).unwrap();
    let mut md = format!("# v4 {phase} summary\n\n- git SHA: `{sha}` (dirty={dirty})\n- hp lock sha256: `{}`\n- seeds: {}\n- manifests: `dataset_manifest.jsonl` (sha256 of file: `{}`)\n\n| Exp | PASS | FAIL | n |\n|---|---|---|---|\n", hp_lock_hash(), seeds.len(), cdt_rqm_epr::v4_dataset::sha256_hex(manifests.as_bytes()));
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
