//! Post-hoc diagnostic (NOT a gate): accuracy on TEST of the consolidated
//! learning signal used *directly* as a predictor ("teacher"), vs A and B.
//! Answers: is B bounded by / distilling the CDT regularity model?
use cdt_rqm_epr::field_autonomy_v4::*;
use cdt_rqm_epr::v4_dataset::*;
use cdt_rqm_epr::v4_metrics as m;
fn main() {
    let phase = std::env::args().nth(1).unwrap_or_else(|| "dev".into());
    let seeds = if phase == "confirm" {
        CONFIRM_SEEDS_V4
    } else {
        DEV_SEEDS_V4
    };
    println!("seed,teacher_acc,teacher_rel_err");
    for s in seeds {
        let sealed = generate_and_seal(s, &DatasetConfig::standard(K_POINTS));
        let cdt = collect_experience(&sealed.ds.train, ExpMode::Real, s);
        let sig = consolidate(&cdt, ConsolidateMode::Full);
        let (mut ok, mut re) = (0usize, vec![]);
        for e in &sealed.ds.test {
            let r = sig.rules.iter().find(|r| r.fam == e.family_id).unwrap();
            let y = rule_apply(r, e.params, &e.input, false);
            ok += m::correct(&y, &e.expected) as usize;
            re.push(m::rel_err(&y, &e.expected));
        }
        println!(
            "0x{s:X},{:.4},{:.4}",
            ok as f64 / sealed.ds.test.len() as f64,
            m::mean(&re)
        );
    }
}
