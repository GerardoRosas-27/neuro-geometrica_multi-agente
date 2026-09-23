fn main() {
    use cdt_rqm_epr::field_autonomy_stage2_v2::{confirmation_deferred_note, run_smoke, DEV_SEEDS};
    let seed = DEV_SEEDS[0];
    println!("Clean-Room v2 smoke seed=0x{seed:X}");
    for row in run_smoke(seed) {
        println!(
            "{}: verdict={} dyn={:.3} static={:.3} leak={} lock={} contam={} notes={}",
            row.experiment,
            row.verdict,
            row.cosine_dynamic,
            row.cosine_static,
            row.leakage_score,
            row.hyperparam_lock,
            row.contamination,
            row.notes
        );
    }
    println!("{}", confirmation_deferred_note());
}
