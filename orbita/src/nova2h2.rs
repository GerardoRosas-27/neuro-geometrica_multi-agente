//! ÓRBITA-Nova-2 · H2 NOVA_MIX (docs/orbita_nova2/PREREG_H2.md).
//!   nova2-h2 --phase dev|confirm|smoke --weights W --seed S --tau T --out DIR [--n-stream N --h-per-fam H --consolidate-every E]
//! Biblioteca C6 legada construida por el flujo NOVA de exp3. NOVA_RET: puntúa todas las entradas por error
//! en k pasos (rollout libre), top-4, elige la mejor; sin ajuste por episodio. Núcleo ÓRBITA congelado.
#![allow(clippy::too_many_arguments)]
use crate::model::Orbita;
use crate::nova::{arbitrate, gate_nodamage, lr_step, OperatorLib, DEFAULT_RANK};
use crate::nova2::err_c3;
use crate::nova_train::{
    ctx_for_eval, fit_c4, gen_heldout_n, gen_stream, k_fit_val, log_rel, lora_prior, med_f64, BufItem,
    Ep, CKPTS, CONSOLIDATE_EVERY, EPS_SPEC, H_PER_FAM, PRIMARY_NEW, STREAM_N,
};
use cdt_rqm_epr::v4_dataset::N_FAM;
use rand::seq::SliceRandom;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::io::Write;
use std::path::PathBuf;

use crate::latent::fit_gn;
use crate::model::M;
use crate::nova2h1::{consolidate_mirror, nova_eval, rank, ret_eval};
const R: usize = 2;
pub const TOPK: usize = 4;
pub const LAM_W: f64 = 1e-2;
const ARMS: [&str; 5] = ["orbita_c3", "nova", "nova_ret", "nova_mix", "nova_mix_shuf"];

/// θ de rango R·n: U=[w_i U_i], V=[V_i].
pub fn mix_th(lib: &OperatorLib, js: &[usize], w: &[f64]) -> Vec<f64> {
    let n = js.len();
    let rr = R * n;
    let mut th = vec![0.0; 2 * rr * M];
    for (b, &j) in js.iter().enumerate() {
        let t = &lib.entries[j].th;
        for s in 0..R { for i in 0..M {
            th[(b * R + s) * M + i] = w[b] * t[s * M + i];
            th[rr * M + (b * R + s) * M + i] = t[R * M + s * M + i];
        } }
    }
    th
}

pub struct MixOut { pub err: f64, pub ok: bool, pub accept: bool, pub pick: Option<usize>, pub w: Vec<f64> }

fn mix_eval(d: &Orbita, e: &Ep, cg: &[f64], k: usize, lib: &OperatorLib, cand: &[usize]) -> MixOut {
    let base = |z: &[f64]| d.step(z, cg);
    if cand.is_empty() {
        let (err, ok) = err_c3(d, cg, e, k, &base);
        return MixOut { err, ok, accept: false, pick: None, w: vec![] };
    }
    let cp = d.ctx(cg);
    let js: Vec<usize> = rank(d, &cp, cg, lib, cand, e, k).into_iter().take(TOPK).map(|x| x.0).collect();
    let n = js.len();
    let mut w0 = vec![0.0; n];
    w0[0] = 1.0;
    let w = fit_gn(&|w, z| lr_step(d, &cp, cg, &mix_th(lib, &js, w), R * n, z), &w0, LAM_W, &e.x0, &e.ys, k);
    let th = mix_th(lib, &js, &w);
    let (kf, kv) = k_fit_val(k);
    let ver = arbitrate(d, &cp, cg, &th, R * n, &e.x0, &e.ys, kf, kv, EPS_SPEC);
    let accept = ver.accept && gate_nodamage(ver.val_err, ver.base_val);
    let ada = |z: &[f64]| lr_step(d, &cp, cg, &th, R * n, z);
    let st: &dyn Fn(&[f64]) -> Vec<f64> = if accept { &ada } else { &base };
    let (err, ok) = err_c3(d, cg, e, k, st);
    MixOut { err, ok, accept, pick: Some(js[0]), w }
}

fn aggregate(d: &Orbita, held: &[Ep], lib: &OperatorLib, lfam: &[usize], tau: f64, seed: u64, n: usize, k: usize) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let all: Vec<usize> = (0..lib.entries.len()).collect();
    let na = ARMS.len();
    let mut cols: Vec<(Vec<f64>, Vec<f64>, usize, usize, usize, usize)> = vec![(vec![], vec![], 0, 0, 0, 0); na];
    // diag por brazo nova_ret / nova_mix / nova_mix_shuf: [acc_new, acc_known, pure_new, pure_known]
    let mut dg = [[0usize; 4]; 3];
    let mut ws: Vec<Vec<f64>> = vec![];
    let mut wshuf: Vec<Vec<f64>> = vec![];
    for (hi, e) in held.iter().enumerate() {
        let isnew = PRIMARY_NEW.contains(&e.fam);
        if !(isnew || e.fam < N_FAM) { continue; }
        let (cg, _, _) = ctx_for_eval(d, e, k);
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x5EF1 ^ ((n as u64) << 20) ^ ((k as u64) << 16) ^ hi as u64);
        let mut sh = all.clone();
        sh.shuffle(&mut rng);
        sh.truncate(TOPK);
        for (ai, name) in ARMS.iter().enumerate() {
            let mut note = |di: usize, acc: bool, pick: Option<usize>| {
                if pick.is_none() { return; }
                let pu = pick.and_then(|j| lfam.get(j).copied()) == Some(e.fam);
                let o = if isnew { 0 } else { 1 };
                dg[di][o] += acc as usize; dg[di][2 + o] += pu as usize;
            };
            let (err, ok) = match *name {
                "orbita_c3" => err_c3(d, &cg, e, k, &|z| d.step(z, &cg)),
                "nova" => nova_eval(d, e, &cg, k, lib, tau, seed),
                "nova_ret" => { let r = ret_eval(d, e, &cg, k, lib, &all); note(0, r.accept, r.pick); (r.err, r.ok) }
                "nova_mix" => { let r = mix_eval(d, e, &cg, k, lib, &all); note(1, r.accept, r.pick); if r.pick.is_some() { ws.push(r.w.clone()); } (r.err, r.ok) }
                _ => { let r = mix_eval(d, e, &cg, k, lib, &sh); note(2, r.accept, r.pick); if r.pick.is_some() { wshuf.push(r.w.clone()); } (r.err, r.ok) }
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
    let wstat = |v: &[Vec<f64>]| {
        let full: Vec<&Vec<f64>> = v.iter().filter(|w| w.len() == TOPK).collect();
        let nf = full.len().max(1) as f64;
        let mut mean = vec![0.0; TOPK];
        let (mut l1, mut share, mut dist, mut far) = (0.0, 0.0, 0.0, 0usize);
        for w in &full {
            for i in 0..TOPK { mean[i] += w[i] / nf; }
            let a: f64 = w.iter().map(|x| x.abs()).sum();
            l1 += a / nf;
            share += w.iter().map(|x| x.abs()).fold(0.0, f64::max) / a.max(1e-300) / nf;
            let dd = ((w[0] - 1.0).powi(2) + w[1..].iter().map(|x| x * x).sum::<f64>()).sqrt();
            dist += dd / nf; far += (dd > 0.1) as usize;
        }
        serde_json::json!({"n": full.len(), "w_mean": mean, "l1_mean": l1, "maxshare_mean": share, "dist_onehot_mean": dist, "frac_dist_gt_0.1": far as f64 / nf})
    };
    let names = ["ret", "mix", "mixshuf"];
    let mut diag = serde_json::Map::new();
    for (i, nm) in names.iter().enumerate() {
        diag.insert(format!("{nm}_accept_new"), (dg[i][0] as f64 / nn).into());
        diag.insert(format!("{nm}_accept_known"), (dg[i][1] as f64 / nk).into());
        diag.insert(format!("{nm}_pick_pure_new"), (dg[i][2] as f64 / nn).into());
        diag.insert(format!("{nm}_pick_pure_known"), (dg[i][3] as f64 / nk).into());
    }
    diag.insert("w_mix".into(), wstat(&ws));
    diag.insert("w_mixshuf".into(), wstat(&wshuf));
    out.insert("diag".into(), serde_json::Value::Object(diag));
    serde_json::Value::Object(out)
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
        "rank": R, "topk": TOPK, "ckpts": rows, "arms": ARMS, "prereg": "docs/orbita_nova2/PREREG_H2.md", "lam_w": LAM_W,
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
    fn topk_const() { assert_eq!(TOPK, 4); assert_eq!(ARMS.len(), 5); }
    #[test]
    fn mix_is_weighted_sum_of_dk() {
        use crate::nova::LibEntry;
        use crate::nova2p0::dk_of;
        let mk = |o: usize| (0..2 * R * M).map(|i| (((i + o) * 37 % 11) as f64 - 5.0) * 0.01).collect::<Vec<f64>>();
        let mut lib = OperatorLib::new();
        for o in 0..4 { lib.entries.push(LibEntry { signature: vec![], th: mk(o), n_support: 1 }); }
        let js = [2usize, 0, 3, 1];
        let w = [0.7, -0.2, 0.4, 0.1];
        let got = dk_of(&mix_th(&lib, &js, &w), R * 4);
        let mut want = vec![0.0; M * M];
        for (b, &j) in js.iter().enumerate() { for (a, x) in want.iter_mut().zip(dk_of(&lib.entries[j].th, R)) { *a += w[b] * x; } }
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-12));
        let one = dk_of(&mix_th(&lib, &js, &[1.0, 0.0, 0.0, 0.0]), R * 4);
        assert!(one.iter().zip(dk_of(&lib.entries[2].th, R)).all(|(a, b)| (a - b).abs() < 1e-12));
    }
}
