//! ÓRBITA-Nova-2 · Exp2 — base compartida A,B + código c∈ℝ² (docs/orbita_nova2/PREREG_EXP2.md).
//!   nova2-exp2 --phase dev|confirm --weights W --seed S --tau T --out DIR [--smoke --n-stream N --h-per-fam H --consolidate-every N]
//! Mantiene los arreglos del exp1 (C3 activo en NOVA*, LAT_EP con τ_LAT(k)). Núcleo ÓRBITA congelado.
#![allow(clippy::too_many_arguments)]
use crate::latent::{ctx8, fit_gn, lat_step, npar};
use crate::model::{Orbita, M};
use crate::nova::{arbitrate, gate_nodamage, lr_step, OperatorLib, DEFAULT_RANK};
use crate::nova2::{err_c3, lat_tau, s_known};
use crate::nova_train::{
    consolidate, ctx_for_eval, err_h16, fit_c4, gen_heldout_n, gen_stream, k_fit_val, log_rel, lora_prior, med_f64,
    BufItem, Ep, CKPTS, CONSOLIDATE_EVERY, EPS_SPEC, H_PER_FAM, LAM_C4, LAT_LAM0, PRIMARY_NEW, STREAM_N,
};
use cdt_rqm_epr::v4_dataset::N_FAM;
use std::io::Write;
use std::path::PathBuf;

const R: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm { Orbita, OrbitaC3, LatEp, Nova, NovaNoC3, NovaBase, NovaBaseNoSparse }
impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Arm::Orbita => "orbita", Arm::OrbitaC3 => "orbita_c3", Arm::LatEp => "lat_ep", Arm::Nova => "nova",
            Arm::NovaNoC3 => "nova_m_c3", Arm::NovaBase => "nova_base", Arm::NovaBaseNoSparse => "nova_base_m_sparse",
        }
    }
    pub fn all() -> &'static [Arm] {
        &[Arm::Orbita, Arm::OrbitaC3, Arm::LatEp, Arm::Nova, Arm::NovaNoC3, Arm::NovaBase, Arm::NovaBaseNoSparse]
    }
}

pub struct PoolItem { pub sig: Vec<f64>, pub th: Vec<f64>, pub known: bool }

#[derive(Clone)]
pub struct Base { pub a: Vec<[f64; M]>, pub b: Vec<[f64; M]>, pub cbar: Vec<f64>, pub n_used: usize }

/// ΔK (M×M, fila i, col j) = Σ_s U_s[i] V_s[j]
fn dk_of(th: &[f64]) -> [[f64; M]; M] {
    let mut k = [[0.0; M]; M];
    for s in 0..R { for i in 0..M { for j in 0..M { k[i][j] += th[s * M + i] * th[R * M + s * M + j]; } } }
    k
}

/// 2 autovectores principales de S simétrica (potencia + deflación, 300 iter., inicio determinista, signo fijado).
fn top2(mut s: [[f64; M]; M]) -> Vec<[f64; M]> {
    let mut out = vec![];
    for r in 0..R {
        let mut x = [0.0; M];
        for i in 0..M { x[i] = 1.0 + 0.1 * ((i + r) as f64); }
        let mut lam = 0.0;
        for _ in 0..300 {
            let mut y = [0.0; M];
            for i in 0..M { for j in 0..M { y[i] += s[i][j] * x[j]; } }
            let n = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            if n < 1e-300 { break; }
            lam = n;
            for i in 0..M { x[i] = y[i] / n; }
        }
        let imax = (0..M).max_by(|&a, &b| x[a].abs().total_cmp(&x[b].abs())).unwrap();
        if x[imax] < 0.0 { for v in x.iter_mut() { *v = -*v; } }
        for i in 0..M { for j in 0..M { s[i][j] -= lam * x[i] * x[j]; } }
        out.push(x);
    }
    out
}

pub fn build_base(items: &[&PoolItem]) -> Option<Base> {
    if items.is_empty() { return None; }
    let (mut sa, mut sb) = ([[0.0; M]; M], [[0.0; M]; M]);
    let dks: Vec<_> = items.iter().map(|p| dk_of(&p.th)).collect();
    for k in &dks {
        for i in 0..M { for j in 0..M { for l in 0..M {
            sa[i][j] += k[i][l] * k[j][l];
            sb[i][j] += k[l][i] * k[l][j];
        } } }
    }
    let (a, b) = (top2(sa), top2(sb));
    let mut cbar = vec![0.0; R];
    for k in &dks {
        for s in 0..R {
            let mut v = 0.0;
            for i in 0..M { for j in 0..M { v += a[s][i] * k[i][j] * b[s][j]; } }
            cbar[s] += v / dks.len() as f64;
        }
    }
    Some(Base { a, b, cbar, n_used: items.len() })
}

/// θ (U,V) de A diag(c) Bᵀ.
pub fn th_of(base: &Base, c: &[f64]) -> Vec<f64> {
    let mut th = vec![0.0; 2 * R * M];
    for s in 0..R { for i in 0..M { th[s * M + i] = base.a[s][i] * c[s]; th[R * M + s * M + i] = base.b[s][i]; } }
    th
}

/// selección sparse: clusters greedy por firma (τ) del pool; sólo clusters con fracción de conocidas < 0,5.
fn select_sparse(pool: &[PoolItem], tau: f64) -> Vec<&PoolItem> {
    let mut cents: Vec<Vec<f64>> = vec![];
    let mut members: Vec<Vec<usize>> = vec![];
    for (i, p) in pool.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (ci, c) in cents.iter().enumerate() {
            let d = OperatorLib::sig_dist(&p.sig, c);
            if d <= tau && best.map_or(true, |(_, bd)| d < bd) { best = Some((ci, d)); }
        }
        if let Some((ci, _)) = best {
            members[ci].push(i);
            let n = members[ci].len() as f64;
            for (a, b) in cents[ci].iter_mut().zip(&p.sig) { *a = (*a * (n - 1.0) + b) / n; }
        } else {
            cents.push(p.sig.clone());
            members.push(vec![i]);
        }
    }
    let mut out = vec![];
    for m in &members {
        let fk = m.iter().filter(|&&i| pool[i].known).count() as f64 / m.len() as f64;
        if fk < 0.5 { out.extend(m.iter().map(|&i| &pool[i])); }
    }
    out
}

pub struct Bases { pub sparse: Option<Base>, pub full: Option<Base> }
fn rebuild(pool: &[PoolItem], tau: f64) -> Bases {
    Bases { sparse: build_base(&select_sparse(pool, tau)), full: build_base(&pool.iter().collect::<Vec<_>>()) }
}

fn eval_arm(d: &Orbita, e: &Ep, k: usize, arm: Arm, lib: &OperatorLib, bases: &Bases, tau: f64, seed: u64) -> (f64, bool) {
    let (cg, gs, _) = ctx_for_eval(d, e, k);
    let (kf, kv) = k_fit_val(k);
    let cp = d.ctx(&cg);
    let base_step = |z: &[f64]| d.step(z, &cg);
    match arm {
        Arm::Orbita => err_h16(&base_step, e),
        Arm::OrbitaC3 => err_c3(d, &cg, e, k, &base_step),
        Arm::LatEp => {
            if s_known(d, &cg, e, k) <= lat_tau(k) {
                err_h16(&base_step, e)
            } else {
                let cx = ctx8(d, e.p);
                let aa = fit_gn(&|th, z| lat_step(d, &cx.cps[gs], &cx.cs[gs], th, z), &vec![0.0; npar()], LAT_LAM0, &e.x0, &e.ys, k);
                err_h16(&|z| lat_step(d, &cx.cps[gs], &cx.cs[gs], &aa, z), e)
            }
        }
        Arm::Nova | Arm::NovaNoC3 => {
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
            if arm == Arm::Nova { err_c3(d, &cg, e, k, step) } else { err_h16(step, e) }
        }
        Arm::NovaBase | Arm::NovaBaseNoSparse => {
            let b = if arm == Arm::NovaBase { &bases.sparse } else { &bases.full };
            let Some(b) = b else { return err_c3(d, &cg, e, k, &base_step) };
            let c = fit_gn(&|c, z| lr_step(d, &cp, &cg, &th_of(b, c), R, z), &b.cbar, LAM_C4, &e.x0, &e.ys, k);
            let th = th_of(b, &c);
            let ver = arbitrate(d, &cp, &cg, &th, R, &e.x0, &e.ys, kf, kv, EPS_SPEC);
            let accept = ver.accept && gate_nodamage(ver.val_err, ver.base_val);
            let ada = |z: &[f64]| lr_step(d, &cp, &cg, &th, R, z);
            let step: &dyn Fn(&[f64]) -> Vec<f64> = if accept { &ada } else { &base_step };
            err_c3(d, &cg, e, k, step)
        }
    }
}

fn aggregate(d: &Orbita, held: &[Ep], lib: &OperatorLib, bases: &Bases, tau: f64, seed: u64, k: usize) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for &arm in Arm::all() {
        let (mut ln, mut lk, mut an, mut ak, mut nn, mut nk) = (vec![], vec![], 0usize, 0usize, 0usize, 0usize);
        for e in held {
            let (err, ok) = eval_arm(d, e, k, arm, lib, bases, tau, seed);
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
    if done.exists() { eprintln!("skip 0x{seed:X}"); return; }
    assert_eq!(DEFAULT_RANK, R);
    let stream = gen_stream(seed, n_stream);
    let held = gen_heldout_n(seed, hpf);
    let mut lib = OperatorLib::new();
    let mut buf: Vec<BufItem> = vec![];
    let mut pool: Vec<PoolItem> = vec![];
    let mut bases = Bases { sparse: None, full: None };
    let (mut acc, mut seen) = (0usize, 0usize);
    let mut ck: Vec<usize> = CKPTS.iter().copied().filter(|&c| c <= n_stream).collect();
    if !ck.contains(&n_stream) { ck.push(n_stream); }
    let t0 = std::time::Instant::now();
    let mut rows = vec![];
    let mut evalc = |n: usize, lib: &OperatorLib, bases: &Bases, pool: &[PoolItem], acc: usize, seen: usize| {
        let r = serde_json::json!({
            "n": n, "lib_size": lib.entries.len(), "pool_size": pool.len(),
            "pool_known": pool.iter().filter(|p| p.known).count(),
            "base_sparse_n": bases.sparse.as_ref().map_or(0, |b| b.n_used),
            "base_full_n": bases.full.as_ref().map_or(0, |b| b.n_used),
            "cbar_sparse": bases.sparse.as_ref().map(|b| b.cbar.clone()),
            "cbar_full": bases.full.as_ref().map(|b| b.cbar.clone()),
            "stream_accept_nova": acc, "stream_seen": seen,
            "k2": aggregate(&d, &held, lib, bases, tau, seed, 2),
            "k4": aggregate(&d, &held, lib, bases, tau, seed, 4),
            "k8": aggregate(&d, &held, lib, bases, tau, seed, 8),
        });
        eprintln!("0x{seed:X} ckpt n={n} lib={} pool={} t={:.0}s", lib.entries.len(), pool.len(), t0.elapsed().as_secs_f64());
        rows.push(r);
    };
    if ck.contains(&0) { evalc(0, &lib, &bases, &pool, acc, seen); }
    for (pos, e) in stream.iter().enumerate() {
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
            pool.push(PoolItem { sig: th0.clone(), th: th.clone(), known: e.fam < N_FAM });
            buf.push(BufItem { signature: th0, th, _fam: e.fam, known: e.fam < N_FAM, val_err: ver.val_err, base_err: ver.base_val });
        }
        seen += 1;
        let n = pos + 1;
        let is_ck = ck.contains(&n);
        if (every > 0 && n % every == 0) || is_ck {
            if !buf.is_empty() { consolidate(&mut lib, &mut buf, tau); }
            bases = rebuild(&pool, tau);
        }
        if is_ck { evalc(n, &lib, &bases, &pool, acc, seen); }
    }
    let rep = serde_json::json!({
        "seed": format!("0x{seed:X}"), "phase": phase, "n_stream": n_stream, "tau": tau, "rank": R,
        "lat_tau": {"2": lat_tau(2), "4": lat_tau(4), "8": lat_tau(8)},
        "ckpts": rows, "prereg": "docs/orbita_nova2/PREREG_EXP2.md",
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
    fn base_recovers_rank2() {
        // ΔK = 3 e0 e1ᵀ + 1 e2 e3ᵀ → A≈{e0,e2}, B≈{e1,e3}, c≈(3,1); th_of reconstruye ΔK.
        let mut th = vec![0.0; 2 * R * M];
        th[0] = 3.0; th[R * M + 1] = 1.0; th[M + 2] = 1.0; th[R * M + M + 3] = 1.0;
        let p = PoolItem { sig: vec![0.0; 4], th: th.clone(), known: false };
        let b = build_base(&[&p]).unwrap();
        assert!((b.cbar[0] - 3.0).abs() < 1e-9 && (b.cbar[1] - 1.0).abs() < 1e-9, "{:?}", b.cbar);
        let k1 = dk_of(&th); let k2 = dk_of(&th_of(&b, &b.cbar));
        for i in 0..M { for j in 0..M { assert!((k1[i][j] - k2[i][j]).abs() < 1e-9); } }
    }
    #[test]
    fn sparse_drops_known_clusters() {
        let mk = |s: f64, kn: bool| PoolItem { sig: vec![s], th: vec![0.0; 2 * R * M], known: kn };
        let pool = vec![mk(0.0, true), mk(0.1, true), mk(10.0, false), mk(10.1, false), mk(10.2, true)];
        assert_eq!(select_sparse(&pool, 1.0).len(), 3);
    }
}
