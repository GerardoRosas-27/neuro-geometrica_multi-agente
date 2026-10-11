//! Follow-up 2: v3.7 E23 rollout variants (diagnostic, DEV seeds 0xA300–0xA30F).
use cdt_rqm_epr::field_autonomy_stage2_v2::{e23_variants, HyperparamLock};
use rayon::prelude::*;
fn main() {
    let mut hp = HyperparamLock::long();
    hp.lock();
    let seeds: Vec<u64> = (0xA300u64..=0xA30F).collect();
    let rows: Vec<Vec<String>> = seeds.par_iter().map(|&s| e23_variants(s, &hp)).collect();
    let mut csv =
        String::from("seed,variant,cos_h1,cos_h2,cos_h4,cos_h8,cos_h16,cos_h32,cos_h64\n");
    for r in rows {
        for l in r {
            csv.push_str(&l);
            csv.push('\n');
        }
    }
    std::fs::create_dir_all("artifacts/v4/E23_variants").unwrap();
    std::fs::write("artifacts/v4/E23_variants/cos_by_horizon.csv", csv).unwrap();
}
