//! Item-1 tuning of the generic consolidation on FRESH tuning seeds
//! 0xC400–0xC407, scored on the DEV partition only (never TEST, never
//! 0xA400/0xB400). Prints teacher DEV accuracy per (D, len, lam).
use cdt_rqm_epr::field_autonomy_v4::*;
use cdt_rqm_epr::v4_dataset::*;
use cdt_rqm_epr::v4_metrics as m;
fn main() {
    GENERIC_CONSOLIDATION.store(true, std::sync::atomic::Ordering::Relaxed);
    for d in ["800", "1600"] {
        for len in ["0.7", "1", "1.4"] {
            for lam in ["1e-8", "1e-7", "1e-6"] {
                std::env::set_var("V4_RFF_D", d);
                std::env::set_var("V4_RFF_LEN", len);
                std::env::set_var("V4_RFF_LAM", lam);
                let mut accs = vec![];
                for s in 0xC400u64..0xC408 {
                    let sealed = generate_and_seal(s, &DatasetConfig::standard(K_POINTS));
                    let sig = consolidate(
                        &collect_experience(&sealed.ds.train, ExpMode::Real, s),
                        ConsolidateMode::Full,
                    );
                    let ok = sealed
                        .ds
                        .dev
                        .iter()
                        .filter(|e| {
                            m::correct(
                                &sig.predict(e.family_id, e.params, &e.input).unwrap(),
                                &e.expected,
                            )
                        })
                        .count();
                    accs.push(ok as f64 / sealed.ds.dev.len() as f64);
                }
                println!(
                    "D={d} len={len} lam={lam} dev_teacher_acc={:.4}",
                    m::mean(&accs)
                );
            }
        }
    }
}
