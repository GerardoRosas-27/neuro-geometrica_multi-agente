//! Design diagnostic for §16 (NOT a gate). Tuning seeds 0xC500.. only,
//! evaluated on the DEV partition (TEST never touched).
use cdt_rqm_epr::field_autonomy_v4::*;
use cdt_rqm_epr::v4_controls::*;
use cdt_rqm_epr::v4_dataset::*;
use cdt_rqm_epr::v4_distill::*;
use cdt_rqm_epr::v4_metrics as m;
use rand::Rng;
use rayon::prelude::*;
fn fam_err(f: &dyn Fn(&Example) -> Vec<f64>, xs: &[Example], h: usize) -> (Vec<f64>, f64) {
    let mut v = vec![vec![]; N_FAM];
    for e in xs {
        v[e.family_id].push(m::rel_err(&f(e), &target_h(e, h)).min(10.0));
    }
    let all: Vec<f64> = v.iter().flatten().cloned().collect();
    (v.iter().map(|x| m::mean(x)).collect(), m::mean(&all))
}
fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let variants: Vec<String> = std::env::args()
        .nth(2)
        .unwrap_or("C1,TRUE,G_C1,G_TRUE".into())
        .split(',')
        .map(String::from)
        .collect();
    let out: Vec<String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let seed = 0xC500 + i;
            let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS));
            let aud = Auditor::new(&s);
            let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
            let sig = consolidate(&cdt, ConsolidateMode::Full);
            let mut lines = vec![];
            let (tf, ta) = fam_err(
                &|e: &Example| sig.predict(e.family_id, e.params, &e.input).unwrap(),
                &s.ds.dev,
                1,
            );
            lines.push(format!(
                "0x{seed:X} TEACHER all={ta:.4} fam={:?}",
                tf.iter()
                    .map(|x| (x * 1e4).round() / 1e4)
                    .collect::<Vec<_>>()
            ));
            let tr = triples(&s.ds.train);
            for v in &variants {
                let gated = v.starts_with("G");
                let hid = if v.contains("136") {
                    136
                } else if v.contains("272") {
                    272
                } else {
                    64
                };
                let mut d = if gated {
                    Dphi::gated_h(12, CTX_DIM, hid, true, seed ^ 0xD0)
                } else {
                    Dphi::with_hidden(12, CTX_DIM, hid, seed ^ 0xD0)
                };
                if v.contains("TRUST") {
                    let (_, t) = train_onestep_trusted(&mut d, &s, seed, &aud, TRUST_TAU);
                    lines.push(format!("0x{seed:X} trusted={t:?}"));
                } else if v.ends_with("TRUE") {
                    d.train_onestep(onestep_budget(), seed ^ 0x7A, ONESTEP_LR_FINAL, true, |r| {
                        tr[r.gen_range(0..tr.len())].clone()
                    });
                } else {
                    train_onestep_route(&mut d, &s, seed, &aud);
                }
                let (f1, a1) = fam_err(&|e: &Example| d.step(&e.input, &e.context), &s.ds.dev, 1);
                let it: Vec<Example> =
                    s.ds.dev
                        .iter()
                        .filter(|e| e.family_id == 1 || e.family_id == 2)
                        .cloned()
                        .collect();
                let hs: Vec<String> = [1usize, 2, 4, 8]
                    .iter()
                    .map(|&h| {
                        let acc = it
                            .iter()
                            .filter(|e| {
                                m::correct(&d.rollout(&e.input, &e.context, h), &target_h(e, h))
                            })
                            .count() as f64
                            / it.len() as f64;
                        let accp = it
                            .iter()
                            .filter(|e| {
                                let mut z = e.input.clone();
                                for _ in 0..h {
                                    z = cdt_rqm_epr::v4_stability::step_variant(
                                        &d, &z, &e.context, &e.input, 1,
                                    )
                                    .0;
                                }
                                m::correct(&z, &target_h(e, h))
                            })
                            .count() as f64
                            / it.len() as f64;
                        format!("h{h}={acc:.2}/P{accp:.2}")
                    })
                    .collect();
                lines.push(format!(
                    "0x{seed:X} {v} all={a1:.4} fam={:?} iter[{}]",
                    f1.iter()
                        .map(|x| (x * 1e4).round() / 1e4)
                        .collect::<Vec<_>>(),
                    hs.join(",")
                ));
            }
            lines.join("\n")
        })
        .collect();
    println!("{}", out.join("\n"));
}
