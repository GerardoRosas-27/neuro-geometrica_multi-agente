//! ÓRBITA-Nova-2 · Exp1 — C4 bayesiano (docs/orbita_nova2/PREREG_EXP1.md).
//!   nova2-eval --phase dev|confirm --weights W --seed S --tau T --out DIR [--smoke --n-stream N --h-per-fam H]
//! Base corregida (diagnóstico §0): C3 activo dentro de NOVA; LAT_EP con τ_LAT(k) propio.
//! Núcleo ÓRBITA congelado.
#![allow(clippy::too_many_arguments)]
use crate::latent::{fit_gn, lat_step, npar, ctx8};
use crate::learn::solve;
use crate::model::{CtxParams, Orbita};
use crate::nova::{arbitrate, gate_nodamage, lr_step, npar_lr, spectral_delta, InnovFilter, OperatorLib, DEFAULT_RANK};
use crate::nova_train::{
    consolidate, ctx_for_eval, err_h16, fin, fit_c4, gen_heldout_n, gen_stream, k_fit_val, log_rel, lora_prior,
    med_f64, BufItem, Ep, CKPTS, CONSOLIDATE_EVERY, EPS_SPEC, H_PER_FAM, LAM_C4, LAT_LAM0, PRIMARY_NEW, STREAM_N,
};
use cdt_rqm_epr::v4_dataset::N_FAM;
use cdt_rqm_epr::v4_metrics as m;
use std::io::Write;
use std::path::PathBuf;

/// τ_LAT(k) congelado del ciclo latente (artifacts/orbita/latente_largo_plazo/tau.json, P95 S_known H DEV fam 1–2).
pub fn lat_tau(k: usize) -> f64 {
    match k {
        2 => 0.014655674499999998,
        4 => 0.0220202535,
        _ => 0.037142499499999995,
    }
}
/// Precisión de prior cuando C6 recupera una entrada (= λm congelado de LAT, selected.json).
pub const LAM_LIB: f64 = 1e-3;
pub const SIG2_FLOOR: f64 = 1e-10;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm { Orbita, OrbitaC3, LatEp, Nova, NovaNoC3, NovaBayes, NovaBayesNoC6 }
impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Arm::Orbita => "orbita", Arm::OrbitaC3 => "orbita_c3", Arm::LatEp => "lat_ep", Arm::Nova => "nova",
            Arm::NovaNoC3 => "nova_m_c3", Arm::NovaBayes => "nova_bayes", Arm::NovaBayesNoC6 => "nova_bayes_m_c6",
        }
    }
    pub fn all() -> &'static [Arm] {
        &[Arm::Orbita, Arm::OrbitaC3, Arm::LatEp, Arm::Nova, Arm::NovaNoC3, Arm::NovaBayes, Arm::NovaBayesNoC6]
    }
}

/// sorpresa base: media rel_err de ÓRBITA en los k pasos observados (S_known de latent.rs).
fn s_known(d: &Orbita, cg: &[f64], e: &Ep, k: usize) -> f64 {
    let mut z = e.x0.clone();
    let mut r = 0.0;
    for t in 0..k {
        z = d.step(&z, cg);
        r += if fin(&z) { m::rel_err(&z, &e.ys[t]).min(10.0) } else { 10.0 };
    }
    r / k as f64
}

/// C3: filtra el estado en los k pasos observados (filtro con operador base) y hace free-run con `step`
/// los 16−k pasos restantes; error relativo en h16 (mismo esquema que ÓRBITA+C3 de Nova).
fn err_c3(d: &Orbita, cg: &[f64], e: &Ep, k: usize, step: &dyn Fn(&[f64]) -> Vec<f64>) -> (f64, bool) {
    let mut filt = InnovFilter::new(1e-4);
    let mut x = e.x0.clone();
    for t in 0..k.min(e.ys.len()) {
        let (xn, _, _) = filt.update(d, cg, &x, &e.ys[t]);
        x = xn;
    }
    let mut z = x;
    for _ in k..16 {
        z = step(&z);
    }
    let err = if fin(&z) { m::rel_err(&z, &e.ys[15]).min(10.0) } else { 10.0 };
    (err, err < 0.05)
}

/// residuos normalizados de rollout desde x0 (igual que fit_gn) para k pasos; None si diverge.
fn resid(step: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], ys: &[Vec<f64>], k: usize) -> Option<(Vec<f64>, Vec<usize>)> {
    let mut z = x0.to_vec();
    let mut r = vec![];
    let mut tt = vec![];
    for t in 0..k {
        z = step(&z);
        if !fin(&z) || z.iter().any(|v| v.abs() > 1e6) {
            return None;
        }
        let n = m::norm(&ys[t]).max(1e-9);
        for (a, b) in z.iter().zip(&ys[t]) {
            r.push((a - b) / n);
            tt.push(t);
        }
    }
    Some((r, tt))
}

pub struct BayesOut { pub th: Vec<f64>, pub accept: bool, pub ll_ada: f64, pub ll_base: f64, pub spectral: f64 }

/// C4 bayesiano (Laplace / GN cerrado) + C5 por verosimilitud predictiva en pasos retenidos.
/// Prior N(μ, σ²/λ I); verosimilitud N(r, σ² I) con σ² = MSE de ÓRBITA en pasos de ajuste (empírico, fijo por episodio).
/// 1) posterior con k_fit pasos → predictiva en k_val pasos retenidos: N(r_v(θ̂), σ² + J_v P⁻¹ J_vᵀ diag).
/// 2) acepta si ll_ada > ll_base (ÓRBITA, var σ²) y espectral ≤ 1+ε. 3) si acepta, θ final = posterior MAP con k pasos.
pub fn bayes_c4(d: &Orbita, cp: &CtxParams, cg: &[f64], e: &Ep, mu: &[f64], lam: f64, rank: usize, k: usize) -> BayesOut {
    let (kf, kv) = k_fit_val(k);
    let stepth = |th: &[f64], z: &[f64]| lr_step(d, cp, cg, th, rank, z);
    let zero = vec![0.0; npar_lr(rank)];
    let base_full = resid(&|z| d.step(z, cg), &e.x0, &e.ys, k);
    let Some((rb, tb)) = base_full else {
        return BayesOut { th: zero, accept: false, ll_ada: f64::NEG_INFINITY, ll_base: f64::NEG_INFINITY, spectral: 0.0 };
    };
    let fit_n: Vec<f64> = rb.iter().zip(&tb).filter(|(_, t)| **t < kf).map(|(r, _)| r * r).collect();
    let sig2 = (fit_n.iter().sum::<f64>() / fit_n.len().max(1) as f64).max(SIG2_FLOOR);
    let th = fit_gn(&stepth, mu, lam, &e.x0, &e.ys, kf);
    let np = th.len();
    let ll_base: f64 = rb.iter().zip(&tb).filter(|(_, t)| **t >= kf).map(|(r, _)| -0.5 * (r * r / sig2 + sig2.ln())).sum();
    let Some((ra, ta)) = resid(&|z| stepth(&th, z), &e.x0, &e.ys, k) else {
        return BayesOut { th, accept: false, ll_ada: f64::NEG_INFINITY, ll_base, spectral: 0.0 };
    };
    // Jacobiano numérico sobre los k pasos
    let nr = ra.len();
    let mut jt = vec![0.0; np * nr];
    for p in 0..np {
        let mut t2 = th.clone();
        let h = 1e-6 * (1.0 + th[p].abs());
        t2[p] += h;
        if let Some((r2, _)) = resid(&|z| stepth(&t2, z), &e.x0, &e.ys, k) {
            for i in 0..nr {
                jt[p * nr + i] = (r2[i] - ra[i]) / h;
            }
        }
    }
    // P = (J_fitᵀ J_fit + λ I)/σ²
    let mut pm = vec![0.0; np * np];
    for p in 0..np {
        for q in 0..=p {
            let s: f64 = (0..nr).filter(|&i| ta[i] < kf).map(|i| jt[p * nr + i] * jt[q * nr + i]).sum();
            pm[p * np + q] = s / sig2;
            pm[q * np + p] = s / sig2;
        }
        pm[p * np + p] += lam / sig2;
    }
    let mut ll_ada = 0.0;
    for i in 0..nr {
        if ta[i] < kf {
            continue;
        }
        let ji: Vec<f64> = (0..np).map(|p| jt[p * nr + i]).collect();
        let x = solve(&pm, &ji, np);
        let s: f64 = ji.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>().max(0.0);
        let v = sig2 + s;
        ll_ada += -0.5 * (ra[i] * ra[i] / v + v.ln());
    }
    let spectral = spectral_delta(&th, rank);
    let accept = spectral <= 1.0 + EPS_SPEC && ll_ada.is_finite() && ll_ada > ll_base + 1e-12;
    let th_final = if accept && kv > 0 { fit_gn(&stepth, mu, lam, &e.x0, &e.ys, k) } else { th };
    BayesOut { th: th_final, accept, ll_ada, ll_base, spectral }
}

/// prior C6 para Bayes: (μ, λ). Con entrada recuperada → (θ_lib, LAM_LIB); si no → (prior0, LAM_C4).
fn bayes_prior(lib: Option<&OperatorLib>, sig: &[f64], tau: f64, prior0: &[f64]) -> (Vec<f64>, f64, bool) {
    if let Some(l) = lib {
        if let Some(ent) = l.retrieve(sig, tau) {
            return (ent.th.clone(), LAM_LIB, true);
        }
    }
    (prior0.to_vec(), LAM_C4, false)
}

fn eval_arm(d: &Orbita, e: &Ep, k: usize, arm: Arm, lib: &OperatorLib, lib_b: &OperatorLib, tau: f64, rank: usize, seed: u64) -> (f64, bool) {
    let (cg, gs, _) = ctx_for_eval(d, e, k);
    let (kf, kv) = k_fit_val(k);
    let cp = d.ctx(&cg);
    match arm {
        Arm::Orbita => err_h16(&|z| d.step(z, &cg), e),
        Arm::OrbitaC3 => err_c3(d, &cg, e, k, &|z| d.step(z, &cg)),
        Arm::LatEp => {
            let sk = s_known(d, &cg, e, k);
            if sk <= lat_tau(k) {
                err_h16(&|z| d.step(z, &cg), e)
            } else {
                let cx = ctx8(d, e.p);
                let aa = fit_gn(&|th, z| lat_step(d, &cx.cps[gs], &cx.cs[gs], th, z), &vec![0.0; npar()], LAT_LAM0, &e.x0, &e.ys, k);
                err_h16(&|z| lat_step(d, &cx.cps[gs], &cx.cs[gs], &aa, z), e)
            }
        }
        Arm::Nova | Arm::NovaNoC3 => {
            let prior0 = lora_prior(seed, e.fam as u64, rank);
            let mut ec = e.clone();
            ec.c = cg.clone();
            let th0 = fit_c4(d, &ec, &prior0, rank, k);
            let prior = lib.retrieve(&th0, tau).map(|en| en.th.clone()).unwrap_or(prior0);
            let th = fit_c4(d, &ec, &prior, rank, k);
            let ver = arbitrate(d, &cp, &cg, &th, rank, &e.x0, &e.ys, kf, kv, EPS_SPEC);
            let accept = ver.accept && gate_nodamage(ver.val_err, ver.base_val);
            let ada = |z: &[f64]| lr_step(d, &cp, &cg, &th, rank, z);
            let base = |z: &[f64]| d.step(z, &cg);
            let step: &dyn Fn(&[f64]) -> Vec<f64> = if accept { &ada } else { &base };
            if arm == Arm::Nova { err_c3(d, &cg, e, k, step) } else { err_h16(step, e) }
        }
        Arm::NovaBayes | Arm::NovaBayesNoC6 => {
            let prior0 = lora_prior(seed, e.fam as u64, rank);
            let mut ec = e.clone();
            ec.c = cg.clone();
            let th0 = fit_c4(d, &ec, &prior0, rank, k);
            let l = if arm == Arm::NovaBayes { Some(lib_b) } else { None };
            let (mu, lam, _) = bayes_prior(l, &th0, tau, &prior0);
            let b = bayes_c4(d, &cp, &cg, e, &mu, lam, rank, k);
            let ada = |z: &[f64]| lr_step(d, &cp, &cg, &b.th, rank, z);
            let base = |z: &[f64]| d.step(z, &cg);
            let step: &dyn Fn(&[f64]) -> Vec<f64> = if b.accept { &ada } else { &base };
            err_c3(d, &cg, e, k, step)
        }
    }
}

fn aggregate(d: &Orbita, held: &[Ep], lib: &OperatorLib, lib_b: &OperatorLib, tau: f64, rank: usize, seed: u64, k: usize) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for &arm in Arm::all() {
        let (mut ln, mut lk, mut an, mut ak, mut nn, mut nk) = (vec![], vec![], 0usize, 0usize, 0usize, 0usize);
        for e in held {
            let (err, ok) = eval_arm(d, e, k, arm, lib, lib_b, tau, rank, seed);
            if PRIMARY_NEW.contains(&e.fam) {
                ln.push(log_rel(err)); nn += 1; an += ok as usize;
            } else if e.fam < N_FAM {
                lk.push(log_rel(err)); nk += 1; ak += ok as usize;
            }
        }
        out.insert(arm.name().into(), serde_json::json!({
            "new_med_log": med_f64(&mut ln), "new_acc16": an as f64 / nn.max(1) as f64,
            "known_med_log": med_f64(&mut lk), "known_acc16": ak as f64 / nk.max(1) as f64,
            "n_new": nn, "n_known": nk,
        }));
    }
    serde_json::Value::Object(out)
}

fn stream_step(d: &Orbita, e: &Ep, lib: &mut OperatorLib, buf: &mut Vec<BufItem>, lib_b: &mut OperatorLib, buf_b: &mut Vec<BufItem>,
               tau: f64, rank: usize, seed: u64, pos: usize, every: usize, stats: &mut [usize; 4]) {
    let k = e.k;
    let (cg, _, _) = ctx_for_eval(d, e, k);
    let (kf, kv) = k_fit_val(k);
    let cp = d.ctx(&cg);
    let mut ec = e.clone();
    ec.c = cg.clone();
    let prior0 = lora_prior(seed, (pos as u64) << 8 ^ e.fam as u64, rank);
    let th0 = fit_c4(d, &ec, &prior0, rank, k);
    // NOVA (idéntico a Nova f64b289)
    let prior = lib.retrieve(&th0, tau).map(|en| en.th.clone()).unwrap_or_else(|| prior0.clone());
    let th = fit_c4(d, &ec, &prior, rank, k);
    let ver = arbitrate(d, &cp, &cg, &th, rank, &e.x0, &e.ys, kf, kv, EPS_SPEC);
    if ver.accept && gate_nodamage(ver.val_err, ver.base_val) {
        stats[0] += 1;
        buf.push(BufItem { signature: th0.clone(), th, _fam: e.fam, known: e.fam < N_FAM, val_err: ver.val_err, base_err: ver.base_val });
    }
    // NOVA_BAYES (biblioteca propia; misma firma/recuperación/consolidación)
    let (mu, lam, hit) = bayes_prior(Some(lib_b), &th0, tau, &prior0);
    stats[2] += hit as usize;
    let b = bayes_c4(d, &cp, &cg, e, &mu, lam, rank, k);
    if b.accept {
        stats[1] += 1;
        buf_b.push(BufItem { signature: th0, th: b.th, _fam: e.fam, known: e.fam < N_FAM, val_err: -b.ll_ada, base_err: -b.ll_base });
    }
    stats[3] += 1;
    if every > 0 && (pos + 1) % every == 0 {
        consolidate(lib, buf, tau);
        consolidate(lib_b, buf_b, tau);
    }
}

fn arg_of(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

pub fn run(a: &[String]) {
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
    if done.exists() {
        eprintln!("skip 0x{seed:X}");
        return;
    }
    let rank = DEFAULT_RANK;
    let stream = gen_stream(seed, n_stream);
    let held = gen_heldout_n(seed, hpf);
    let (mut lib, mut lib_b) = (OperatorLib::new(), OperatorLib::new());
    let (mut buf, mut buf_b) = (vec![], vec![]);
    let mut stats = [0usize; 4];
    let mut rows = vec![];
    let mut ck: Vec<usize> = CKPTS.iter().copied().filter(|&c| c <= n_stream).collect();
    if !ck.contains(&n_stream) { ck.push(n_stream); }
    let t0 = std::time::Instant::now();
    let mut evalc = |n: usize, lib: &OperatorLib, lib_b: &OperatorLib, stats: &[usize; 4]| {
        let r = serde_json::json!({
            "n": n, "lib_size": lib.entries.len(), "lib_b_size": lib_b.entries.len(),
            "stream_accept_nova": stats[0], "stream_accept_bayes": stats[1], "stream_hit_bayes": stats[2], "stream_seen": stats[3],
            "k2": aggregate(&d, &held, lib, lib_b, tau, rank, seed, 2),
            "k4": aggregate(&d, &held, lib, lib_b, tau, rank, seed, 4),
            "k8": aggregate(&d, &held, lib, lib_b, tau, rank, seed, 8),
        });
        eprintln!("0x{seed:X} ckpt n={n} lib={} lib_b={} t={:.0}s", lib.entries.len(), lib_b.entries.len(), t0.elapsed().as_secs_f64());
        rows.push(r);
    };
    if ck.contains(&0) { evalc(0, &lib, &lib_b, &stats); }
    for (pos, e) in stream.iter().enumerate() {
        stream_step(&d, e, &mut lib, &mut buf, &mut lib_b, &mut buf_b, tau, rank, seed, pos, every, &mut stats);
        let n = pos + 1;
        if ck.contains(&n) {
            if !buf.is_empty() { consolidate(&mut lib, &mut buf, tau); }
            if !buf_b.is_empty() { consolidate(&mut lib_b, &mut buf_b, tau); }
            evalc(n, &lib, &lib_b, &stats);
        }
    }
    let rep = serde_json::json!({
        "seed": format!("0x{seed:X}"), "phase": phase, "n_stream": n_stream, "tau": tau, "rank": rank,
        "lat_tau": {"2": lat_tau(2), "4": lat_tau(4), "8": lat_tau(8)}, "lam_lib": LAM_LIB,
        "ckpts": rows, "prereg": "docs/orbita_nova2/PREREG_EXP1.md",
        "arms": Arm::all().iter().map(|a| a.name()).collect::<Vec<_>>(), "smoke": smoke,
    });
    let mut f = std::fs::File::create(out_dir.join(format!("metrics_0x{seed:X}.json"))).unwrap();
    writeln!(f, "{}", serde_json::to_string_pretty(&rep).unwrap()).unwrap();
    assert_eq!(Orbita::from_json(&wtxt).raw, before, "núcleo ÓRBITA modificado");
    std::fs::write(&done, b"ok\n").unwrap();
}
