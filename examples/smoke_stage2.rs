fn main() {
    use cdt_rqm_epr::field_autonomy_stage2::{
        run_e18a, run_e18b, run_e19, run_e20, run_fase_a_scaffold,
    };
    let seed = 0xE1800u64;
    for (name, row) in [
        ("FASE_A", run_fase_a_scaffold(seed)),
        ("E18A", run_e18a(seed)),
        ("E18B", run_e18b(seed)),
        ("E19", run_e19(seed)),
        ("E20", run_e20(seed)),
    ] {
        println!(
            "{name}: verdict={} dyn={:.3} static={:.3} leak={} notes={}",
            row.verdict, row.cosine_dynamic, row.cosine_static, row.leakage_score, row.notes
        );
    }
}
