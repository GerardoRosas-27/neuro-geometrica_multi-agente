//! ÓRBITA-Nova-2 · Exp3 — BOCD sustituye el umbral fijo de novedad (docs/orbita_nova2/PREREG_EXP3.md).
//!   nova2-exp3-calib --weights-dir DIR --out FILE        (una vez; DEV fam 1–2)
//!   nova2-exp3 --phase dev|confirm --weights W --seed S --tau T --out DIR [--smoke --n-stream N --h-per-fam H --consolidate-every N]
//! Mantiene C3 activo en NOVA* y LAT_EP con τ_LAT(k). Núcleo ÓRBITA congelado.
#![allow(clippy::too_many_arguments)]
use crate::latent::{ctx8, fit_gn, lat_step, npar};
use crate::model::Orbita;
use crate::nova::{arbitrate, gate_nodamage, lr_step, OperatorLib, DEFAULT_RANK};
use crate::nova2::{err_c3, lat_tau, s_known};
use crate::nova_train::{
    consolidate, ctx_for_eval, err_h16, fin, fit_c4, gen_calib_h, gen_heldout_n, gen_stream, k_fit_val, log_rel,
    lora_prior, med_f64, BufItem, Ep, CKPTS, CONSOLIDATE_EVERY, EPS_SPEC, H_PER_FAM, LAT_LAM0, PRIMARY_NEW, STREAM_N,
};
use cdt_rqm_epr::v4_dataset::N_FAM;
use cdt_rqm_epr::v4_metrics as m;
use rand::seq::SliceRandom;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::io::Write;
use std::path::PathBuf;

const R: usize = 2;
/// PREREG_EXP3 §2.2–2.3 (h, umbral a priori; μ_ref, s²_ref de nova2-exp3-calib, DEV fam 1–2).
pub const HAZARD: f64 = 1.0 / 16.0;
pub const P_THR: f64 = 0.5;
pub const MU_REF: f64 = -5.301001033444018; // calib.json
pub const S2_REF: f64 = 0.489366781028144; // calib.json
const N0_REF: f64 = 16.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm { Orbita, OrbitaC3, LatEp, Nova, NovaNoC3, LatBocd, NovaBocd, LatBocdShuf }
impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Arm::Orbita => "orbita", Arm::OrbitaC3 => "orbita_c3", Arm::LatEp => "lat_ep", Arm::Nova => "nova",
            Arm::NovaNoC3 => "nova_m_c3", Arm::LatBocd => "lat_bocd", Arm::NovaBocd => "nova_bocd",
            Arm::LatBocdShuf => "lat_bocd_shuf",
        }
    }
    pub fn all() -> &'static [Arm] {
        &[Arm::Orbita, Arm::OrbitaC3, Arm::LatEp, Arm::Nova, Arm::NovaNoC3, Arm::LatBocd, Arm::NovaBocd, Arm::LatBocdShuf]
    }
}

/// Normal-Gamma (μ, κ, α, β).
#[derive(Clone, Copy, Debug)]
pub struct Ng { pub mu: f64, pub kappa: f64, pub alpha: f64, pub beta: f64 }
impl Ng {
    pub fn ln_pred(&self, x: f64) -> f64 {
        let nu = 2.0 * self.alpha;
        let s2 = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa);
        let z = (x - self.mu) * (x - self.mu) / (nu * s2);
        ln_gamma((nu + 1.0) / 2.0) - ln_gamma(nu / 2.0) - 0.5 * (nu * std::f64::consts::PI * s2).ln()
            - (nu + 1.0) / 2.0 * (1.0 + z).ln()
    }
    pub fn update(&self, x: f64) -> Ng {
        let k1 = self.kappa + 1.0;
        Ng {
            mu: (self.kappa * self.mu + x) / k1, kappa: k1, alpha: self.alpha + 0.5,
            beta: self.beta + self.kappa * (x - self.mu) * (x - self.mu) / (2.0 * k1),
        }
    }
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos (g=7, n=9)
    const G: [f64; 9] = [0.999_999_999_999_809_9, 676.5203681218851, -1259.1392167224028, 771.323_428_777_653_1,
        -176.615_029_162_140_6, 12.507343278686905, -0.13857109526572012, 9.984_369_578_019_572e-6, 1.5056327351493116e-7];
    if x < 0.5 { return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x); }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) { a += g / (x + i as f64); }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

pub fn ng_ref0() -> Ng { Ng { mu: MU_REF, kappa: N0_REF, alpha: N0_REF / 2.0, beta: N0_REF / 2.0 * S2_REF } }
pub fn ng_new0(r: &Ng) -> Ng { let _ = r; Ng { mu: MU_REF, kappa: 0.1, alpha: 1.0, beta: S2_REF } }

/// x_t = ln rel_err one-step de ÓRBITA, t=1..k (PREREG §2.1).
pub fn obs(d: &Orbita, cg: &[f64], e: &Ep, k: usize) -> Vec<f64> {
    let mut prev = e.x0.clone();
    (0..k.min(e.ys.len())).map(|t| {
        let z = d.step(&prev, cg);
        let r = if fin(&z) { m::rel_err(&z, &e.ys[t]).min(10.0) } else { 10.0 };
        prev = e.ys[t].clone();
        log_rel(r)
    }).collect()
}

/// BOCD con run inicial = ref (prob. 1). Devuelve p_cambio = 1 − P(run = ref | x).
pub fn bocd_pchange(ref_ng: &Ng, xs: &[f64]) -> f64 {
    let new0 = ng_new0(ref_ng);
    let mut lref = 0.0f64; // log P(ref)
    let mut rref = *ref_ng;
    let mut runs: Vec<(f64, Ng)> = vec![]; // (log p, stats)
    for &x in xs {
        let lall: Vec<f64> = std::iter::once(lref).chain(runs.iter().map(|r| r.0)).collect();
        let lse_all = lse(&lall);
        let mut nref = lref + (1.0 - HAZARD).ln() + rref.ln_pred(x);
        let mut nr: Vec<(f64, Ng)> = runs.iter().map(|(lp, s)| (lp + (1.0 - HAZARD).ln() + s.ln_pred(x), s.update(x))).collect();
        nr.push((lse_all + HAZARD.ln() + new0.ln_pred(x), new0.update(x)));
        let z = lse(&std::iter::once(nref).chain(nr.iter().map(|r| r.0)).collect::<Vec<_>>());
        nref -= z;
        for r in nr.iter_mut() { r.0 -= z; }
        lref = nref; rref = rref.update(x); runs = nr;
    }
    (1.0 - lref.exp()).clamp(0.0, 1.0)
}

fn lse(v: &[f64]) -> f64 {
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !mx.is_finite() { return mx; }
    mx + v.iter().map(|x| (x - mx).exp()).sum::<f64>().ln()
}

fn auroc(s: &[f64], y: &[bool]) -> f64 {
    let (mut num, mut np, mut nn) = (0.0, 0usize, 0usize);
    for i in 0..s.len() { if y[i] { np += 1 } else { nn += 1 } }
    for i in 0..s.len() { if !y[i] { continue; } for j in 0..s.len() { if y[j] { continue; }
        num += if s[i] > s[j] { 1.0 } else if s[i] == s[j] { 0.5 } else { 0.0 }; } }
    if np == 0 || nn == 0 { f64::NAN } else { num / (np * nn) as f64 }
}

fn lat_err(d: &Orbita, e: &Ep, k: usize, gs: usize) -> (f64, bool) {
    let cx = ctx8(d, e.p);
    let aa = fit_gn(&|th, z| lat_step(d, &cx.cps[gs], &cx.cs[gs], th, z), &vec![0.0; npar()], LAT_LAM0, &e.x0, &e.ys, k);
    err_h16(&|z| lat_step(d, &cx.cps[gs], &cx.cs[gs], &aa, z), e)
}

/// `novel` = decisión que usa el brazo (BOCD propia, o permutada para SHUF).
fn eval_arm(d: &Orbita, e: &Ep, k: usize, arm: Arm, lib: &OperatorLib, tau: f64, seed: u64, novel: bool) -> (f64, bool) {
    let (cg, gs, _) = ctx_for_eval(d, e, k);
    let (kf, kv) = k_fit_val(k);
    let cp = d.ctx(&cg);
    let base_step = |z: &[f64]| d.step(z, &cg);
    match arm {
        Arm::Orbita => err_h16(&base_step, e),
        Arm::OrbitaC3 => err_c3(d, &cg, e, k, &base_step),
        Arm::LatEp => if s_known(d, &cg, e, k) <= lat_tau(k) { err_h16(&base_step, e) } else { lat_err(d, e, k, gs) },
        Arm::LatBocd | Arm::LatBocdShuf => if !novel { err_h16(&base_step, e) } else { lat_err(d, e, k, gs) },
        Arm::Nova | Arm::NovaNoC3 | Arm::NovaBocd => {
            if arm == Arm::NovaBocd && !novel { return err_c3(d, &cg, e, k, &base_step); }
            let prior0 = lora_prior(seed, e.fam as u64, R);
            let mut ec = e.clone();
            ec.c = cg.clone();
            let th0 = fit_c4(d, &ec, &prior0, R, k);
            let prior = lib.retrieve(&th0, tau).map(|en| en.th.clone()).unwrap_or(prior0);
            let th = fit_c4(d, &ec, &prior, R, k);
            let ver = arbitrate(d, &cp, &cg, &th, R, &e.x0, &e.ys, kf, kv, EPS_SPEC);
            let accept = ver.accept && gate_nodamage(ver.val_err, ver.base_val);
            let ada = |z: &[f64]| lr_step(d, &cp, &cg, &th, R, z);
            let step: &dyn Fn(&[f64]) -> Vec<f64> = if accept { &ada } else { &base_step };
            if arm == Arm::NovaNoC3 { err_h16(step, e) } else { err_c3(d, &cg, e, k, step) }
        }
    }
}

fn aggregate(d: &Orbita, held: &[Ep], lib: &OperatorLib, rf: &Ng, tau: f64, seed: u64, n: usize, k: usize) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let mut pch = vec![]; let mut sk = vec![]; let mut lab = vec![]; let mut keep = vec![];
    for e in held {
        let (cg, _, _) = ctx_for_eval(d, e, k);
        pch.push(bocd_pchange(rf, &obs(d, &cg, e, k)));
        sk.push(s_known(d, &cg, e, k));
        lab.push(PRIMARY_NEW.contains(&e.fam));
        keep.push(PRIMARY_NEW.contains(&e.fam) || e.fam < N_FAM);
    }
    let dec: Vec<bool> = pch.iter().map(|&p| p > P_THR).collect();
    let mut shuf = dec.clone();
    shuf.shuffle(&mut Xoshiro256StarStar::seed_from_u64(seed ^ 0xB0CD ^ n as u64 ^ k as u64));
    // detección (sólo nuevas {8,10,11} vs conocidas 0–7)
    let idx: Vec<usize> = (0..held.len()).filter(|&i| keep[i]).collect();
    let y: Vec<bool> = idx.iter().map(|&i| lab[i]).collect();
    let bacc = |pred: &dyn Fn(usize) -> bool| {
        let tp = idx.iter().filter(|&&i| lab[i] && pred(i)).count() as f64 / idx.iter().filter(|&&i| lab[i]).count().max(1) as f64;
        let tn = idx.iter().filter(|&&i| !lab[i] && !pred(i)).count() as f64 / idx.iter().filter(|&&i| !lab[i]).count().max(1) as f64;
        (0.5 * (tp + tn), 1.0 - tn)
    };
    let (bb, fpb) = bacc(&|i| dec[i]);
    let (bt, fpt) = bacc(&|i| sk[i] > lat_tau(k));
    out.insert("detect".into(), serde_json::json!({
        "bacc_bocd": bb, "fpr_bocd": fpb, "bacc_tau": bt, "fpr_tau": fpt,
        "auroc_bocd": auroc(&idx.iter().map(|&i| pch[i]).collect::<Vec<_>>(), &y),
        "auroc_sknown": auroc(&idx.iter().map(|&i| sk[i]).collect::<Vec<_>>(), &y),
    }));
    for &arm in Arm::all() {
        let (mut ln, mut lk, mut an, mut ak, mut nn, mut nk) = (vec![], vec![], 0usize, 0usize, 0usize, 0usize);
        for (i, e) in held.iter().enumerate() {
            if !keep[i] { continue; }
            let nov = if arm == Arm::LatBocdShuf { shuf[i] } else { dec[i] };
            let (err, ok) = eval_arm(d, e, k, arm, lib, tau, seed, nov);
            if lab[i] { ln.push(log_rel(err)); nn += 1; an += ok as usize; } else { lk.push(log_rel(err)); nk += 1; ak += ok as usize; }
        }
        out.insert(arm.name().into(), serde_json::json!({
            "new_med_log": med_f64(&mut ln), "new_acc16": an as f64 / nn.max(1) as f64,
            "known_med_log": med_f64(&mut lk), "known_acc16": ak as f64 / nk.max(1) as f64,
            "n_new": nn, "n_known": nk,
        }));
    }
    serde_json::Value::Object(out)
}

fn arg_of(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

/// Calibración única (PREREG §2.3): μ, s² de x_t (pasos 1..8) en gen_calib_h(seed,16), fam 1–2, 16 seeds DEV.
pub fn run_calib(a: &[String]) {
    let wd = arg_of(a, "--weights-dir").expect("--weights-dir");
    let mut xs = vec![];
    for s in 0xA400u64..=0xA40F {
        let d = Orbita::from_json(&std::fs::read_to_string(format!("{wd}/weights_0x{s:X}.json")).unwrap());
        for e in gen_calib_h(s, 16) {
            assert!(e.fam == 1 || e.fam == 2);
            let (cg, _, _) = ctx_for_eval(&d, &e, 8);
            xs.extend(obs(&d, &cg, &e, 8));
        }
    }
    let n = xs.len() as f64;
    let mu = xs.iter().sum::<f64>() / n;
    let s2 = xs.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / (n - 1.0);
    let j = serde_json::json!({"mu_ref": mu, "s2_ref": s2, "n_obs": xs.len(), "seeds": "0xA400-0xA40F", "fams": [1,2],
        "per_seed_eps": 32, "steps": 8, "prereg": "docs/orbita_nova2/PREREG_EXP3.md §2.3"});
    let out = arg_of(a, "--out").expect("--out");
    if let Some(p) = PathBuf::from(&out).parent() { std::fs::create_dir_all(p).unwrap(); }
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("{j}");
}

pub fn run(a: &[String]) {
    assert!(MU_REF.is_finite() && S2_REF.is_finite(), "calibración no congelada");
    let wtxt = std::fs::read_to_string(arg_of(a, "--weights").expect("--weights")).unwrap();
    let d = Orbita::from_json(&wtxt);
    let before = d.raw.clone();
    let s = arg_of(a, "--seed").expect("--seed");
    let seed = u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap();
    let tau: f64 = arg_of(a, "--tau").expect("--tau").parse().unwrap();
    let smoke = a.iter().any(|x| x == "--smoke");
    let n_stream: usize = arg_of(a, "--n-stream").map(|s| s.parse().unwrap()).unwrap_or(if smoke { 24 } else { STREAM_N });
    let hpf: usize = arg_of(a, "--h-per-fam").map(|s| s.parse().unwrap()).unwrap_or(if smoke { 2 } else { H_PER_FAM });
    let every: usize = arg_of(a, "--consolidate-every").map(|s| s.parse().unwrap()).unwrap_or(if smoke { 8 } else { CONSOLIDATE_EVERY });
    let phase = arg_of(a, "--phase").unwrap_or_else(|| "dev".into());
    let out_dir = PathBuf::from(arg_of(a, "--out").expect("--out")).join(&phase);
    std::fs::create_dir_all(&out_dir).unwrap();
    let done = out_dir.join(format!("0x{seed:X}.done"));
    if done.exists() { eprintln!("skip 0x{seed:X}"); return; }
    assert_eq!(DEFAULT_RANK, R);
    let stream = gen_stream(seed, n_stream);
    let held = gen_heldout_n(seed, hpf);
    let mut lib = OperatorLib::new();
    let mut buf: Vec<BufItem> = vec![];
    let mut rf = ng_ref0();
    let (mut acc, mut seen, mut s_nov, mut s_nov_known, mut s_nov_new, mut s_known_n, mut s_new_n) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut ck: Vec<usize> = CKPTS.iter().copied().filter(|&c| c <= n_stream).collect();
    if !ck.contains(&n_stream) { ck.push(n_stream); }
    let t0 = std::time::Instant::now();
    let mut rows = vec![];
    let mut evalc = |n: usize, lib: &OperatorLib, rf: &Ng, st: [usize; 7]| {
        let r = serde_json::json!({
            "n": n, "lib_size": lib.entries.len(),
            "ref_ng": {"mu": rf.mu, "kappa": rf.kappa, "alpha": rf.alpha, "beta": rf.beta},
            "stream_accept_nova": st[0], "stream_seen": st[1], "stream_novel": st[2],
            "stream_novel_on_known": st[3], "stream_novel_on_new": st[4], "stream_known": st[5], "stream_new": st[6],
            "k2": aggregate(&d, &held, lib, rf, tau, seed, n, 2),
            "k4": aggregate(&d, &held, lib, rf, tau, seed, n, 4),
            "k8": aggregate(&d, &held, lib, rf, tau, seed, n, 8),
        });
        eprintln!("0x{seed:X} ckpt n={n} lib={} ref_kappa={:.0} t={:.0}s", lib.entries.len(), rf.kappa, t0.elapsed().as_secs_f64());
        rows.push(r);
    };
    if ck.contains(&0) { evalc(0, &lib, &rf, [0; 7]); }
    for (pos, e) in stream.iter().enumerate() {
        let k = e.k;
        let (cg, _, _) = ctx_for_eval(&d, e, k);
        // BOCD en línea (sin etiquetas): sólo episodios decididos «conocidos» actualizan NG_ref.
        let xs = obs(&d, &cg, e, k);
        let isnew = PRIMARY_NEW.contains(&e.fam);
        if isnew { s_new_n += 1 } else if e.fam < N_FAM { s_known_n += 1 }
        if bocd_pchange(&rf, &xs) > P_THR {
            s_nov += 1; if isnew { s_nov_new += 1 } else if e.fam < N_FAM { s_nov_known += 1 }
        } else {
            for &x in &xs { rf = rf.update(x); }
        }
        // flujo NOVA idéntico a exp2
        let (kf, kv) = k_fit_val(k);
        let cp = d.ctx(&cg);
        let mut ec = e.clone();
        ec.c = cg.clone();
        let prior0 = lora_prior(seed, (pos as u64) << 8 ^ e.fam as u64, R);
        let th0 = fit_c4(&d, &ec, &prior0, R, k);
        let prior = lib.retrieve(&th0, tau).map(|en| en.th.clone()).unwrap_or_else(|| prior0.clone());
        let th = fit_c4(&d, &ec, &prior, R, k);
        let ver = arbitrate(&d, &cp, &cg, &th, R, &e.x0, &e.ys, kf, kv, EPS_SPEC);
        if ver.accept && gate_nodamage(ver.val_err, ver.base_val) {
            acc += 1;
            buf.push(BufItem { signature: th0, th, _fam: e.fam, known: e.fam < N_FAM, val_err: ver.val_err, base_err: ver.base_val });
        }
        seen += 1;
        let n = pos + 1;
        let is_ck = ck.contains(&n);
        if ((every > 0 && n % every == 0) || is_ck) && !buf.is_empty() { consolidate(&mut lib, &mut buf, tau); }
        if is_ck { evalc(n, &lib, &rf, [acc, seen, s_nov, s_nov_known, s_nov_new, s_known_n, s_new_n]); }
    }
    let rep = serde_json::json!({
        "seed": format!("0x{seed:X}"), "phase": phase, "n_stream": n_stream, "tau": tau, "rank": R,
        "lat_tau": {"2": lat_tau(2), "4": lat_tau(4), "8": lat_tau(8)},
        "bocd": {"hazard": HAZARD, "p_thr": P_THR, "mu_ref": MU_REF, "s2_ref": S2_REF, "n0_ref": N0_REF},
        "ckpts": rows, "prereg": "docs/orbita_nova2/PREREG_EXP3.md",
        "arms": Arm::all().iter().map(|a| a.name()).collect::<Vec<_>>(), "smoke": smoke,
    });
    let mut f = std::fs::File::create(out_dir.join(format!("metrics_0x{seed:X}.json"))).unwrap();
    writeln!(f, "{}", serde_json::to_string_pretty(&rep).unwrap()).unwrap();
    assert_eq!(Orbita::from_json(&wtxt).raw, before, "núcleo ÓRBITA modificado");
    std::fs::write(&done, b"ok\n").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lngamma_ok() {
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-10);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10);
    }
    #[test]
    fn pred_integrates() {
        let g = Ng { mu: -3.0, kappa: 2.0, alpha: 1.5, beta: 0.7 };
        let s: f64 = (0..40000).map(|i| -23.0 + i as f64 * 0.001).map(|x| g.ln_pred(x).exp() * 0.001).sum();
        assert!((s - 1.0).abs() < 2e-3, "{s}");
    }
    #[test]
    fn bocd_detects_shift() {
        let r = Ng { mu: -4.0, kappa: 16.0, alpha: 8.0, beta: 8.0 * 0.25 };
        // con MU_REF NaN el prior new usa NaN → probamos con la lógica a mano: p_cambio de datos en ref debe ser bajo
        if MU_REF.is_finite() {
            assert!(bocd_pchange(&r, &[-4.0, -4.1]) < 0.5);
            assert!(bocd_pchange(&r, &[0.0, 0.2]) > 0.5);
        }
        assert!(r.update(1.0).kappa == 17.0);
    }
}
