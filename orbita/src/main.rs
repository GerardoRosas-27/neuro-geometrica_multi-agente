//! ÓRBITA runner (separado del código v4; sólo *usa* la librería v4 sin modificarla).
//!   orbita export --seed S --out DIR        datos TRAIN + replay del teacher (CDT→consolidación) + auditoría
//!   orbita eval   --seed S --weights W --out DIR [--label L]   protocolo §16 DIST_E: 1 paso TEST + E33 + proceso nuevo
//!   orbita fresh  --seed S --weights W --test-sha SHA          evaluador en proceso nuevo (E37)
//!   orbita parity --weights W --probe P                        paridad Rust↔PyTorch
mod model;
use cdt_rqm_epr::field_autonomy_v4::*;
use cdt_rqm_epr::v4_dataset::{generate_and_seal, DatasetConfig, Example, Sealed, N_FAM};
use cdt_rqm_epr::v4_distill::{family_fit_residual, onestep_breakdown, TRUST_TAU};
use cdt_rqm_epr::v4_metrics as m;
use cdt_rqm_epr::v4_provenance::{snapshot, QueryCounters};
use model::Orbita;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::io::Write;

pub const POOL: usize = 100_000;
pub const KS: [usize; 4] = [1, 2, 4, 8];

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}
fn seed_of(a: &[String]) -> u64 {
    let s = arg(a, "--seed").expect("--seed");
    if let Some(h) = s.strip_prefix("0x") { u64::from_str_radix(h, 16).unwrap() } else { s.parse().unwrap() }
}
fn sealed(seed: u64) -> Sealed { generate_and_seal(seed, &DatasetConfig::standard(K_POINTS)) }
fn put(f: &mut Vec<u8>, v: &[f64]) { for x in v { f.extend_from_slice(&x.to_le_bytes()); } }
fn fam_of(c: &[f64]) -> (usize, [f64; 2]) {
    let f = (0..N_FAM).max_by(|&a, &b| c[a].total_cmp(&c[b])).unwrap();
    (f, [c[N_FAM], c[N_FAM + 1]])
}

/// Igual que `train_trusted_on` (v4 §16) hasta la máscara de confianza; el replay se
/// materializa en un pool (con objetivos multi-paso del *teacher*, nunca del generador).
fn export(a: &[String]) {
    let seed = seed_of(a);
    let out = std::path::PathBuf::from(arg(a, "--out").expect("--out"));
    std::fs::create_dir_all(&out).unwrap();
    let s = sealed(seed);
    let aud = Auditor::new(&s);
    let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
    let sig = consolidate(&cdt, ConsolidateMode::Full);
    let res = family_fit_residual(&sig, cdt.read_all());
    let w: Vec<f64> = res.iter().map(|r| (-r / TRUST_TAU).exp()).collect();
    let mut tb = vec![];
    for (x, c, y) in triples(&s.ds.train) { put(&mut tb, &x); put(&mut tb, &c); put(&mut tb, &y); }
    std::fs::write(out.join("train.bin"), &tb).unwrap();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x0B17A);
    let mut rb = vec![];
    let mut hits: std::collections::BTreeSet<u64> = Default::default();
    let mut any_global = false;
    for _ in 0..POOL {
        let (x, c, y1) = replay_sample(&sig, &mut rng, K_POINTS);
        let (f, p) = fam_of(&c);
        for h in aud.hits(&y1) { if h == u64::MAX { any_global = true } else { hits.insert(h); } }
        put(&mut rb, &x); put(&mut rb, &c);
        let mut y = x.clone();
        let mut step = 0;
        for &k in &KS {
            while step < k { y = sig.predict(f, p, &y).unwrap(); step += 1; }
            if k == 1 { assert!(cdt_rqm_epr::v4_dataset::l2(&y, &y1) < 1e-12); }
            put(&mut rb, &y);
        }
        put(&mut rb, &[w[f]]);
    }
    std::fs::write(out.join("replay.bin"), &rb).unwrap();
    drop(sig); drop(cdt);
    let meta = serde_json::json!({
        "seed": format!("0x{seed:X}"), "n_train": s.ds.train.len(), "pool": POOL, "ks": KS,
        "zdim": 2 * K_POINTS, "cdim": cdt_rqm_epr::v4_dataset::CTX_DIM,
        "fit_residual": res, "trust_w": w, "trusted_mask": res.iter().map(|r| *r < TRUST_TAU).collect::<Vec<_>>(),
        "replay_equiv_hits": hits.iter().collect::<Vec<_>>(), "replay_any_equiv_global": any_global,
        "test_sha256": s.manifest.test_sha256, "replay_fraction": REPLAY_FRACTION,
    });
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    println!("exported 0x{seed:X}: train={} pool={POOL} mask={:?}", s.ds.train.len(), meta["trusted_mask"]);
}

fn flags_for(s: &Sealed, seed: u64, meta: &serde_json::Value) -> AuditFlags {
    let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
    let mut f = AuditFlags { cdt_canon: cdt.canonical_targets(), ..Default::default() };
    for h in meta["replay_equiv_hits"].as_array().unwrap() { f.equiv_hits.insert(h.as_u64().unwrap()); }
    f.any_equiv_global = meta["replay_any_equiv_global"].as_bool().unwrap();
    f
}

fn row(seed: u64, cond: &str, part: &str, n: usize, params: usize, acc: f64, cos: f64, en: f64, md: f64, st: f64, leak: usize, q: QueryCounters, status: &str, notes: String) -> Row {
    Row { experiment: "ORBITA".into(), seed, condition: cond.into(), partition: part.into(), n_examples: n, params, compute: 0,
        accuracy: acc, cosine: cos, energy: en, manifold_distance: md, stability: st, leakage: leak, q, status: status.into(), notes }
}

/// Copia fiel de `v4_distill::e33_rollout` (privada en v4) para un `Predictor` genérico, variante V0.
fn e33(seed: u64, label: &str, d: &Orbita, iter_ex: &[&Example], deltas: &[Vec<f64>]) -> (usize, Vec<Row>, String) {
    let mut zs: Vec<Vec<f64>> = iter_ex.iter().map(|e| e.input.clone()).collect();
    let mut zp: Vec<Vec<f64>> = iter_ex.iter().zip(deltas).map(|(e, dl)| e.input.iter().zip(dl).map(|(a, b)| a + b).collect()).collect();
    let (mut region, mut contiguous) = (0usize, true);
    let (mut rows, mut notes, mut sig_all) = (vec![], vec![], vec![]);
    for t in 1..=*HORIZONS.last().unwrap() {
        for (i, e) in iter_ex.iter().enumerate() {
            let z = d.step(&zs[i], &e.context);
            sig_all.push(m::sigma_max(&m::jacobian(|w| d.step(w, &e.context), &zs[i])));
            zs[i] = z;
            zp[i] = d.step(&zp[i], &e.context);
        }
        if HORIZONS.contains(&t) {
            let (mut ok, mut re, mut dr, mut md, mut en, mut pg) = (0usize, vec![], vec![], vec![], vec![], vec![]);
            let (mut fam_ok, mut fam_n) = ([0usize; 2], [0usize; 2]);
            for (i, e) in iter_ex.iter().enumerate() {
                let y = target_h(e, t);
                let c = m::correct(&zs[i], &y);
                ok += c as usize;
                let fi = if e.family_id == 1 { 0 } else { 1 };
                fam_n[fi] += 1; fam_ok[fi] += c as usize;
                re.push(m::rel_err(&zs[i], &y).min(10.0));
                dr.push(m::norm(&zs[i]) / m::norm(&y).max(1e-12));
                en.push(m::energy(&zs[i]) / m::energy(&y).max(1e-12));
                md.push(m::manifold_distance(&e.input, &zs[i]));
                pg.push(cdt_rqm_epr::v4_dataset::l2(&zp[i], &zs[i]) / m::norm(&deltas[i]).max(1e-12));
            }
            let acc = ok as f64 / iter_ex.len() as f64;
            if contiguous && acc >= 0.5 { region = t; } else { contiguous = false; }
            let (ar, af) = (fam_ok[0] as f64 / fam_n[0].max(1) as f64, fam_ok[1] as f64 / fam_n[1].max(1) as f64);
            notes.push(format!("h{t}:acc={acc:.3}(rot={ar:.2},refl={af:.2}),rel={:.3},norm={:.3},pert={:.3}", m::mean(&re), m::median(&dr), m::median(&pg)));
            rows.push(row(seed, &format!("{label}_h{t}"), "TEST_iterable", iter_ex.len(), d.n_params(), acc, m::median(&pg), m::median(&en), m::median(&md), m::mean(&re), 0, QueryCounters::default(), "CLEAN",
                format!("norm_ratio={:.4} pert_growth={:.4} acc_rot={ar:.4} acc_refl={af:.4}", m::median(&dr), m::median(&pg))));
        }
    }
    (region, rows, format!("sigma_med={:.4} {}", m::median(&sig_all), notes.join(" ")))
}

fn eval(a: &[String]) {
    let seed = seed_of(a);
    let wpath = arg(a, "--weights").expect("--weights");
    let label = arg(a, "--label").unwrap_or_else(|| "ORBITA".into());
    let out = std::path::PathBuf::from(arg(a, "--out").expect("--out"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(arg(a, "--meta").expect("--meta")).unwrap()).unwrap();
    let d = Orbita::from_json(&std::fs::read_to_string(&wpath).unwrap());
    let s = sealed(seed);
    assert_eq!(meta["test_sha256"].as_str().unwrap(), s.manifest.test_sha256);
    let aud = Auditor::new(&s);
    let flags = flags_for(&s, seed, &meta);
    s.verify_test().expect("TEST sealed");
    let ev = evaluate(&d, &s.ds.test, 1, Some((&aud, &flags)), None);
    let clean = ev.leaked == 0 && ev.queries.total() == 0;
    // proceso nuevo
    let exe = std::env::current_exe().unwrap();
    let o = std::process::Command::new(exe).args(["fresh", "--seed", &format!("{seed}"), "--weights", &wpath, "--test-sha", &s.manifest.test_sha256]).output().unwrap();
    let v: serde_json::Value = serde_json::from_str(String::from_utf8_lossy(&o.stdout).lines().last().unwrap_or("{}")).unwrap_or_default();
    let (fp_acc, fp_q) = (v["acc"].as_f64().unwrap_or(f64::NAN), v["queries_total"].as_f64().unwrap_or(f64::NAN));
    let fresh_ok = (fp_acc - ev.acc).abs() < 1e-12 && fp_q == 0.0;
    let mut rows = vec![row(seed, &label, "TEST", ev.n, d.n_params(), ev.acc, ev.disp_cos, ev.energy_ratio, ev.manifold, ev.rel_err, ev.leaked, ev.queries, if clean { "CLEAN" } else { "LEAKED" },
        format!("mask={} fresh_acc={fp_acc:.6} fresh_q={fp_q} fresh_ok={fresh_ok} {}", meta["trusted_mask"], onestep_breakdown(&d, &s.ds.test)))];
    let iter_ex: Vec<&Example> = s.ds.test.iter().filter(|e| ITERABLE_FAMS.contains(&e.family_id)).collect();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xD33);
    let deltas: Vec<Vec<f64>> = iter_ex.iter().map(|e| {
        let v: Vec<f64> = e.input.iter().map(|_| rng.gen_range(-1.0..1.0)).collect();
        let nv = m::norm(&v).max(1e-12);
        v.iter().map(|x| x / nv * 0.01 * m::norm(&e.input)).collect()
    }).collect();
    let rl = format!("R_{label}_V0");
    let (region, rr, notes) = e33(seed, &rl, &d, &iter_ex, &deltas);
    rows.extend(rr);
    let pass = region >= 8 && clean && fresh_ok;
    rows.push(row(seed, "GATE", &rl, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, QueryCounters::default(), if pass { "PASS" } else { "FAIL" },
        format!("{rl} region={region} rel_err1={:.4} clean={clean} fresh={fp_acc:.4}(q={fp_q}) {notes}", ev.rel_err)));
    std::fs::create_dir_all(&out).unwrap();
    let mut f = std::fs::File::create(out.join(format!("metrics_0x{seed:X}.csv"))).unwrap();
    writeln!(f, "{CSV_HEADER}").unwrap();
    for r in &rows { writeln!(f, "{}", r.csv()).unwrap(); }
    println!("0x{seed:X} {label} acc1={:.4} rel1={:.4} region={region} clean={clean} fresh_ok={fresh_ok} pass={pass}", ev.acc, ev.rel_err);
}

fn fresh(a: &[String]) {
    let seed = seed_of(a);
    let d = Orbita::from_json(&std::fs::read_to_string(arg(a, "--weights").unwrap()).unwrap());
    let s = sealed(seed);
    if arg(a, "--test-sha").as_deref() != Some(s.manifest.test_sha256.as_str()) { println!("{{\"error\":\"test_sha_mismatch\"}}"); return; }
    let before = snapshot();
    let acc = evaluate(&d, &s.ds.test, 1, None, None).acc;
    let q = snapshot().delta(&before).total();
    println!("{{\"acc\":{acc},\"queries_total\":{q},\"pid\":{}}}", std::process::id());
}

fn parity(a: &[String]) {
    let d = Orbita::from_json(&std::fs::read_to_string(arg(a, "--weights").unwrap()).unwrap());
    let p: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(arg(a, "--probe").unwrap()).unwrap()).unwrap();
    let mut worst: f64 = 0.0;
    for case in p.as_array().unwrap() {
        let f = |k: &str| -> Vec<f64> { case[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() };
        let y = d.step(&f("x"), &f("c"));
        worst = worst.max(cdt_rqm_epr::v4_dataset::l2(&y, &f("y")));
    }
    println!("{{\"params\":{},\"max_l2_diff\":{worst:e}}}", d.n_params());
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("export") => export(&a),
        Some("eval") => eval(&a),
        Some("fresh") => fresh(&a),
        Some("parity") => parity(&a),
        Some("bench") => bench(&a),
        _ => eprintln!("uso: orbita export|eval|fresh|parity ..."),
    }
}

/// Benchmark de inferencia (1 hilo): ÓRBITA vs Dφ E3 (ckpt v4) vs M3 (misma arquitectura gated no residual).
fn bench(a: &[String]) {
    use cdt_rqm_epr::v4_controls::{Dphi, Predictor};
    use std::time::Instant;
    let seed = seed_of(a);
    let orb = Orbita::from_json(&std::fs::read_to_string(arg(a, "--weights").unwrap()).unwrap());
    let e3 = Dphi::from_ckpt(&std::fs::read_to_string(arg(a, "--e3").unwrap()).unwrap()).unwrap();
    let m3 = Dphi::gated_h(2 * K_POINTS, cdt_rqm_epr::v4_dataset::CTX_DIM, 136, false, seed ^ 0xD0); // pesos no entrenados: mismo coste
    let s = sealed(seed);
    let ex: Vec<&Example> = s.ds.test.iter().collect();
    let reps: usize = arg(a, "--reps").and_then(|v| v.parse().ok()).unwrap_or(20000);
    let models: Vec<(&str, &dyn Predictor)> = vec![("ORBITA", &orb), ("v4_E3", &e3), ("M3_mlp", &m3)];
    println!("model,params,mean_us_step_b1,p50_us,p95_us,steps_per_s_b1,steps_per_s_b1024,rollout16_mean_us,rollout16_p95_us");
    for (name, mdl) in models {
        let mut sink = 0.0;
        for i in 0..2000 { let e = ex[i % ex.len()]; sink += mdl.predict(&e.input, &e.context)[0]; } // warmup
        let mut lat = Vec::with_capacity(reps);
        for i in 0..reps { let e = ex[i % ex.len()]; let t = Instant::now(); sink += mdl.predict(&e.input, &e.context)[0]; lat.push(t.elapsed().as_nanos() as f64 / 1e3); }
        let mut sl = lat.clone(); sl.sort_by(|x, y| x.total_cmp(y));
        let mean = lat.iter().sum::<f64>() / reps as f64;
        // batch grande: 1024 estados, 20 pasadas
        let big: Vec<&Example> = (0..1024).map(|i| ex[i % ex.len()]).collect();
        let t = Instant::now();
        for _ in 0..20 { for e in &big { sink += mdl.predict(&e.input, &e.context)[0]; } }
        let tp_big = (20 * 1024) as f64 / t.elapsed().as_secs_f64();
        let mut ro = vec![];
        for i in 0..2000 { let e = ex[i % ex.len()]; let t = Instant::now(); let mut z = e.input.clone(); for _ in 0..16 { z = mdl.predict(&z, &e.context); } sink += z[0]; ro.push(t.elapsed().as_nanos() as f64 / 1e3); }
        let rm = ro.iter().sum::<f64>() / ro.len() as f64; ro.sort_by(|x, y| x.total_cmp(y));
        println!("{name},{},{mean:.2},{:.2},{:.2},{:.0},{tp_big:.0},{rm:.1},{:.1}", mdl.params(), sl[reps / 2], sl[reps * 95 / 100], 1e6 / mean, ro[ro.len() * 95 / 100]);
        if sink == 12345.678 { println!("x"); }
    }
}
