//! ÓRBITA-Nova-2 · H1 NOVA_RET (docs/orbita_nova2/PREREG_H1.md).
//!   nova2-h1 --phase dev|confirm|smoke --weights W --seed S --tau T --out DIR [--n-stream N --h-per-fam H --consolidate-every E]
//! Biblioteca C6 legada construida por el flujo NOVA de exp3. NOVA_RET: puntúa todas las entradas por error
//! en k pasos (rollout libre), top-4, elige la mejor; sin ajuste por episodio. Núcleo ÓRBITA congelado.
#![allow(clippy::too_many_arguments)]
use crate::model::Orbita;
use crate::nova::{arbitrate, gate_nodamage, lr_step, OperatorLib, DEFAULT_RANK};
use crate::nova2::err_c3;
use crate::nova_train::{
    consolidate, ctx_for_eval, fin, fit_c4, gen_heldout_n, gen_stream, k_fit_val, log_rel, lora_prior, med_f64, BufItem,
    Ep, CKPTS, CONSOLIDATE_EVERY, EPS_SPEC, H_PER_FAM, PRIMARY_NEW, STREAM_N,
};
use cdt_rqm_epr::v4_dataset::N_FAM;
use cdt_rqm_epr::v4_metrics as m;
use rand::seq::SliceRandom;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::io::Write;
use std::path::PathBuf;

const R: usize = 2;
pub const TOPK: usize = 4;
const ARMS: [&str; 4] = ["orbita_c3", "nova", "nova_ret", "nova_ret_shuf"];

/// s_j = media ln rel_err de rollout libre k pasos (PREREG §1.3).
pub fn kscore(d: &Orbita, cp: &crate::model::CtxParams, cg: &[f64], th: &[f64], e: &Ep, k: usize) -> f64 {
    let mut z = e.x0.clone();
    let kk = k.min(e.ys.len());
    let mut s = 0.0;
    for t in 0..kk {
        z = lr_step(d, cp, cg, th, R, &z);
        let r = if fin(&z) { m::rel_err(&z, &e.ys[t]).min(10.0) } else { 10.0 };
        if !fin(&z) { s += log_rel(10.0) * (kk - t) as f64; return s / kk as f64; }
        s += log_rel(r);
    }
    s / kk.max(1) as f64
}

/// Índices ordenados por puntuación (empates: índice menor) sobre `cand`.
pub fn rank(d: &Orbita, cp: &crate::model::CtxParams, cg: &[f64], lib: &OperatorLib, cand: &[usize], e: &Ep, k: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = cand.iter().map(|&j| (j, kscore(d, cp, cg, &lib.entries[j].th, e, k))).collect();
    v.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    v
}

pub struct RetOut { pub err: f64, pub ok: bool, pub accept: bool, pub pick: Option<usize>, pub score: f64 }

pub fn ret_eval(d: &Orbita, e: &Ep, cg: &[f64], k: usize, lib: &OperatorLib, cand: &[usize]) -> RetOut {
    let base = |z: &[f64]| d.step(z, cg);
    if cand.is_empty() {
        let (err, ok) = err_c3(d, cg, e, k, &base);
        return RetOut { err, ok, accept: false, pick: None, score: f64::NAN };
    }
    let cp = d.ctx(cg);
    let rk = rank(d, &cp, cg, lib, cand, e, k);
    let top: Vec<(usize, f64)> = rk.into_iter().take(TOPK).collect();
    let (j, sc) = top[0]; // mejor del top-4 (= argmin)
    let th = &lib.entries[j].th;
    let (kf, kv) = k_fit_val(k);
    let ver = arbitrate(d, &cp, cg, th, R, &e.x0, &e.ys, kf, kv, EPS_SPEC);
    let accept = ver.accept && gate_nodamage(ver.val_err, ver.base_val);
    let ada = |z: &[f64]| lr_step(d, &cp, cg, th, R, z);
    let st: &dyn Fn(&[f64]) -> Vec<f64> = if accept { &ada } else { &base };
    let (err, ok) = err_c3(d, cg, e, k, st);
    RetOut { err, ok, accept, pick: Some(j), score: sc }
}

pub fn nova_eval(d: &Orbita, e: &Ep, cg: &[f64], k: usize, lib: &OperatorLib, tau: f64, seed: u64) -> (f64, bool) {
    let (kf, kv) = k_fit_val(k);
    let cp = d.ctx(cg);
    let base = |z: &[f64]| d.step(z, cg);
    let prior0 = lora_prior(seed, e.fam as u64, R);
    let mut ec = e.clone();
    ec.c = cg.to_vec();
    let th0 = fit_c4(d, &ec, &prior0, R, k);
    let prior = lib.retrieve(&th0, tau).map(|en| en.th.clone()).unwrap_or(prior0);
    let th = fit_c4(d, &ec, &prior, R, k);
    let ver = arbitrate(d, &cp, cg, &th, R, &e.x0, &e.ys, kf, kv, EPS_SPEC);
    let accept = ver.accept && gate_nodamage(ver.val_err, ver.base_val);
    let ada = |z: &[f64]| lr_step(d, &cp, cg, &th, R, z);
    let st: &dyn Fn(&[f64]) -> Vec<f64> = if accept { &ada } else { &base };
    err_c3(d, cg, e, k, st)
}

fn aggregate(d: &Orbita, held: &[Ep], lib: &OperatorLib, lfam: &[usize], tau: f64, seed: u64, n: usize, k: usize) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let all: Vec<usize> = (0..lib.entries.len()).collect();
    let mut cols: Vec<(Vec<f64>, Vec<f64>, usize, usize, usize, usize)> = vec![(vec![], vec![], 0, 0, 0, 0); 4];
    let (mut acc_n, mut acc_k, mut pure_n, mut pure_k, mut picks, mut sc_n, mut sc_s) = (0usize, 0usize, 0usize, 0usize, vec![], vec![], vec![]);
    for (hi, e) in held.iter().enumerate() {
        let isnew = PRIMARY_NEW.contains(&e.fam);
        if !(isnew || e.fam < N_FAM) { continue; }
        let (cg, _, _) = ctx_for_eval(d, e, k);
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x5EF1 ^ ((n as u64) << 20) ^ ((k as u64) << 16) ^ hi as u64);
        let mut sh = all.clone();
        sh.shuffle(&mut rng);
        sh.truncate(TOPK);
        for (ai, name) in ARMS.iter().enumerate() {
            let (err, ok) = match *name {
                "orbita_c3" => err_c3(d, &cg, e, k, &|z| d.step(z, &cg)),
                "nova" => nova_eval(d, e, &cg, k, lib, tau, seed),
                "nova_ret" => {
                    let r = ret_eval(d, e, &cg, k, lib, &all);
                    if let Some(j) = r.pick {
                        picks.push(j);
                        let pu = lfam.get(j).copied() == Some(e.fam);
                        if isnew { acc_n += r.accept as usize; pure_n += pu as usize; sc_n.push(r.score); }
                        else { acc_k += r.accept as usize; pure_k += pu as usize; }
                    }
                    (r.err, r.ok)
                }
                _ => { let r = ret_eval(d, e, &cg, k, lib, &sh); if isnew && r.pick.is_some() { sc_s.push(r.score); } (r.err, r.ok) }
            };
            let c = &mut cols[ai];
            if isnew { c.0.push(log_rel(err)); c.2 += 1; c.4 += ok as usize; } else { c.1.push(log_rel(err)); c.3 += 1; c.5 += ok as usize; }
        }
    }
    let (nn, nk) = (cols[0].2.max(1) as f64, cols[0].3.max(1) as f64);
    for (ai, name) in ARMS.iter().enumerate() {
        let c = &mut cols[ai];
        out.insert((*name).into(), serde_json::json!({
            "new_med_log": med_f64(&mut c.0), "known_med_log": med_f64(&mut c.1),
            "new_acc16": c.4 as f64 / c.2.max(1) as f64, "known_acc16": c.5 as f64 / c.3.max(1) as f64, "n_new": c.2, "n_known": c.3,
        }));
    }
    picks.sort(); picks.dedup();
    out.insert("diag".into(), serde_json::json!({
        "ret_accept_new": acc_n as f64 / nn, "ret_accept_known": acc_k as f64 / nk,
        "ret_pick_pure_new": pure_n as f64 / nn, "ret_pick_pure_known": pure_k as f64 / nk,
        "ret_distinct_picks": picks.len(), "ret_score_new_med": med_f64(&mut sc_n), "shuf_score_new_med": med_f64(&mut sc_s),
    }));
    serde_json::Value::Object(out)
}

/// Espejo de familias: replica el clustering greedy de `consolidate` (sin alterarlo) y asigna la familia dominante
/// a cada entrada nueva por coincidencia exacta de θ medio.
pub fn consolidate_mirror(lib: &mut OperatorLib, buf: &mut Vec<BufItem>, tau: f64, lfam: &mut Vec<usize>) {
    let mut cl: Vec<Vec<usize>> = vec![];
    let mut ce: Vec<Vec<f64>> = vec![];
    for (i, b) in buf.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (ci, c) in ce.iter().enumerate() {
            let dd = OperatorLib::sig_dist(&b.signature, c);
            if dd <= tau && best.map_or(true, |(_, bd)| dd < bd) { best = Some((ci, dd)); }
        }
        if let Some((ci, _)) = best {
            cl[ci].push(i);
            let n = cl[ci].len() as f64;
            for (a, x) in ce[ci].iter_mut().zip(&buf[i].signature) { *a = (*a * (n - 1.0) + x) / n; }
        } else { cl.push(vec![i]); ce.push(b.signature.clone()); }
    }
    let info: Vec<(Vec<f64>, usize)> = cl.iter().map(|mem| {
        let np = buf[mem[0]].th.len();
        let mut th = vec![0.0; np];
        let mut fc = [0usize; 16];
        for &i in mem { for (a, x) in th.iter_mut().zip(&buf[i].th) { *a += x; } fc[buf[i]._fam.min(15)] += 1; }
        th.iter_mut().for_each(|v| *v /= mem.len() as f64);
        (th, fc.iter().enumerate().max_by_key(|x| (x.1, std::cmp::Reverse(x.0))).map(|x| x.0).unwrap())
    }).collect();
    let l0 = lib.entries.len();
    consolidate(lib, buf, tau);
    for en in &lib.entries[l0..] {
        let f = info.iter().min_by(|a, b| OperatorLib::sig_dist(&a.0, &en.th).total_cmp(&OperatorLib::sig_dist(&b.0, &en.th))).map(|x| x.1).unwrap_or(99);
        lfam.push(f);
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
    let n_stream: usize = arg_of(a, "--n-stream").map(|s| s.parse().unwrap()).unwrap_or(STREAM_N);
    let hpf: usize = arg_of(a, "--h-per-fam").map(|s| s.parse().unwrap()).unwrap_or(H_PER_FAM);
    let every: usize = arg_of(a, "--consolidate-every").map(|s| s.parse().unwrap()).unwrap_or(CONSOLIDATE_EVERY);
    let phase = arg_of(a, "--phase").unwrap_or_else(|| "dev".into());
    let out_dir = PathBuf::from(arg_of(a, "--out").expect("--out")).join(&phase);
    std::fs::create_dir_all(&out_dir).unwrap();
    let done = out_dir.join(format!("0x{seed:X}.done"));
    if done.exists() { eprintln!("skip 0x{seed:X}"); return; }
    assert_eq!(DEFAULT_RANK, R);
    let stream = gen_stream(seed, n_stream);
    let held = gen_heldout_n(seed, hpf);
    let mut lib = OperatorLib::new();
    let mut lfam: Vec<usize> = vec![];
    let mut buf: Vec<BufItem> = vec![];
    let mut acc = 0usize;
    let mut ck: Vec<usize> = CKPTS.iter().copied().filter(|&c| c <= n_stream).collect();
    if !ck.contains(&n_stream) { ck.push(n_stream); }
    let t0 = std::time::Instant::now();
    let mut rows = vec![];
    let mut evalc = |n: usize, lib: &OperatorLib, lfam: &[usize], acc: usize| {
        let r = serde_json::json!({
            "n": n, "lib_size": lib.entries.len(), "lib_dom_fam": lfam, "stream_accept_nova": acc,
            "k2": aggregate(&d, &held, lib, lfam, tau, seed, n, 2),
            "k4": aggregate(&d, &held, lib, lfam, tau, seed, n, 4),
            "k8": aggregate(&d, &held, lib, lfam, tau, seed, n, 8),
        });
        eprintln!("0x{seed:X} ckpt n={n} lib={} t={:.0}s", lib.entries.len(), t0.elapsed().as_secs_f64());
        rows.push(r);
    };
    if ck.contains(&0) { evalc(0, &lib, &lfam, 0); }
    for (pos, e) in stream.iter().enumerate() {
        // flujo NOVA idéntico a exp3
        let k = e.k;
        let (cg, _, _) = ctx_for_eval(&d, e, k);
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
        let n = pos + 1;
        let is_ck = ck.contains(&n);
        if ((every > 0 && n % every == 0) || is_ck) && !buf.is_empty() { consolidate_mirror(&mut lib, &mut buf, tau, &mut lfam); }
        if is_ck { evalc(n, &lib, &lfam, acc); }
    }
    let rep = serde_json::json!({
        "seed": format!("0x{seed:X}"), "phase": phase, "n_stream": n_stream, "h_per_fam": hpf, "every": every, "tau": tau,
        "rank": R, "topk": TOPK, "ckpts": rows, "arms": ARMS, "prereg": "docs/orbita_nova2/PREREG_H1.md",
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
    fn topk_const() { assert_eq!(TOPK, 4); assert_eq!(ARMS.len(), 4); }
}
