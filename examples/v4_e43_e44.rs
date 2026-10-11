//! v4 E43/E44 runner (preregistro §11). Usage: v4_e43_e44 <dev|confirm|smoke>
use cdt_rqm_epr::v4_dataset::sha256_hex;
use cdt_rqm_epr::v4_language::*;
use rayon::prelude::*;
fn main() {
    let ph = std::env::args().nth(1).unwrap_or_else(|| "dev".into());
    let seeds: Vec<u64> = match ph.as_str() {
        "dev" => (0xA400..0xA410).collect(),
        "confirm" => (0xB400..0xB410).collect(),
        _ => vec![0x5A00],
    };
    let ga = load_emb("artifacts/v4/lang/emb_gemma.json").expect("gemma cache");
    let gb = load_emb("artifacts/v4/lang/emb_qwen.json").expect("qwen cache");
    let outs: Vec<(u64, LangOut)> = seeds
        .par_iter()
        .map(|&s| (s, run_seed(s, &ga, &gb)))
        .collect();
    let dir = format!("artifacts/v4/E43_E44/{ph}");
    std::fs::create_dir_all(&dir).unwrap();
    let mut csv = String::from("seed,exp,encoder,lang,subset,method,n,acc\n");
    let mut md = format!("# E43/E44 — {ph}\n\n");
    let sha = |p: &str| sha256_hex(&std::fs::read(p).unwrap());
    let git = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    md.push_str(&format!(
        "- git: `{git}`\n- emb_gemma sha256: `{}`\n- emb_qwen sha256: `{}`\n\n",
        sha("artifacts/v4/lang/emb_gemma.json"),
        sha("artifacts/v4/lang/emb_qwen.json")
    ));
    let p43 = outs.iter().filter(|o| o.1.e43_pass).count();
    let p44 = outs.iter().filter(|o| o.1.e44_pass).count();
    md.push_str(&format!("| exp | PASS | FAIL | n |\n|---|---|---|---|\n| E43 | {p43} | {} | {} |\n| E44 | {p44} | {} | {} |\n\n", outs.len() - p43, outs.len(), outs.len() - p44, outs.len()));
    for (s, o) in &outs {
        for r in &o.rows {
            csv.push_str(r);
            csv.push('\n');
        }
        md.push_str(&format!(
            "- 0x{s:X} E43 {} — {}\n- 0x{s:X} E44 {} — {}\n",
            if o.e43_pass { "PASS" } else { "FAIL" },
            o.note43,
            if o.e44_pass { "PASS" } else { "FAIL" },
            o.note44
        ));
    }
    std::fs::write(format!("{dir}/metrics.csv"), csv).unwrap();
    std::fs::write(format!("{dir}/summary.md"), &md).unwrap();
    println!("{md}");
}
