//! v4 E23 diagnosis (v3.7 LONG lock, DEV seeds 0xA300–0xA30F). Writes
//! artifacts/v4/E23_diag/trace.csv
use cdt_rqm_epr::field_autonomy_stage2_v2::{diagnose_e23, HyperparamLock};
use rayon::prelude::*;
fn main() {
    let mut hp = HyperparamLock::long();
    hp.lock();
    let seeds: Vec<u64> = (0xA300u64..=0xA30F).collect();
    let rows: Vec<Vec<String>> = seeds.par_iter().map(|&s| diagnose_e23(s, &hp)).collect();
    let mut csv = String::from("seed,step,cos_target,raw_norm_pre_projection,sigma_max_J,true_step_cos,true_point_norm,manifold_dist,one_step_err_on_true,pert_growth_eps[0.001;0.01;0.05;0.10;0.20]\n");
    for r in rows {
        for l in r {
            csv.push_str(&l);
            csv.push('\n');
        }
    }
    std::fs::create_dir_all("artifacts/v4/E23_diag").unwrap();
    std::fs::write("artifacts/v4/E23_diag/trace.csv", csv).unwrap();
}
