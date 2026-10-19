//! Adaptación en el espacio latente de Koopman + memoria episódica + flujo largo (docs/prereg_orbita_latente_largo_plazo.md).
//! No lee ds.test; núcleo ÓRBITA congelado (sólo se leen sus pesos).
//!   lat-eval exp1 --weights W --set S --ctx C --k K --tau T --lam0 L0 --lamm LM            → CSV por episodio (stdout)
//!   lat-eval long --weights W --set S --ctx C(stream; cond = kN) --held H --hset HS --taus t2,t4,t8 --lam0 --lamm
//!                 --oep-d --oep-l --oms-d --oms-l --ckpts 0,125,... --held-out OUT        → stream (stdout) + held-out (OUT)
use crate::fam::{ls_fit, ls_step, reid, rule};
use crate::learn::{solve, Module as Res};
use crate::model::{CtxParams, Orbita, M};
use cdt_rqm_epr::v4_dataset::{context_of, N_FAM};
use cdt_rqm_epr::v4_metrics as m;

pub const DEG: usize = 2; // grado de los monomios extra del lifting latente
pub const TOPM: usize = 8; // vecinos de la memoria para el prior
const RES_DEG: usize = 2; const RES_LAM: f64 = 1e-4; // adaptador residual de e2ac4ab (fijo)
pub const OEP_GRID: [(usize, f64); 10] = [(2, 1e-4), (2, 1e-3), (2, 1e-2), (2, 1e-1), (2, 1.0), (3, 1e-4), (3, 1e-3), (3, 1e-2), (3, 1e-1), (3, 1.0)];
pub const OMS_GRID: [(usize, f64); 8] = [(2, 1e-4), (2, 1e-3), (2, 1e-2), (2, 1e-1), (3, 1e-4), (3, 1e-3), (3, 1e-2), (3, 1e-1)];

fn fin(z: &[f64]) -> bool { z.iter().all(|v| v.is_finite()) }
fn nums(s: &str, sep: char) -> Vec<f64> { s.split(sep).filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect() }
fn mono(x: f64, y: f64, lo: usize, hi: usize) -> Vec<f64> { let mut v = vec![]; for t in lo..=hi { for j in 0..=t { v.push(x.powi((t - j) as i32) * y.powi(j as i32)); } } v }
pub fn nxi() -> usize { M + mono(0.0, 0.0, 2, DEG).len() + 1 }
pub fn npar() -> usize { nxi() * M }

/// contextos ÓRBITA precalculados para las 8 familias base a p̂ del episodio
pub struct Ctx8 { pub cps: Vec<CtxParams>, pub cs: Vec<Vec<f64>> }
pub fn ctx8(d: &Orbita, p: [f64; 2]) -> Ctx8 { let cs: Vec<Vec<f64>> = (0..N_FAM).map(|g| context_of(g, p)).collect(); Ctx8 { cps: cs.iter().map(|c| d.ctx(c)).collect(), cs } }
/// LAT: por punto q: z = enc(q; c), u = K(c) z + Aᵀ ξ(q), ξ = [z, monomios grado 2..DEG, 1]; q' = Brenier(u)[0..2]
pub fn lat_step(d: &Orbita, cp: &CtxParams, c: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for i in 0..x.len() / 2 {
        let z = d.encode(&x[2 * i..2 * i + 2], c); let mut u = d.core(cp, &z);
        let mut xi = z.clone(); xi.extend(mono(x[2 * i], x[2 * i + 1], 2, DEG)); xi.push(1.0);
        for (r, v) in xi.iter().enumerate() { for j in 0..M { u[j] += v * a[r * M + j]; } }
        let w = d.correct(cp, &u); out[2 * i] = w[0]; out[2 * i + 1] = w[1];
    }
    out
}
/// oráculo directo multipaso: q' = q + Wᵀ φ_{0..D}(q)
fn dir_step(w: &[f64], deg: usize, x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for i in 0..x.len() / 2 { let f = mono(x[2 * i], x[2 * i + 1], 0, deg); for (r, v) in f.iter().enumerate() { y[2 * i] += v * w[2 * r]; y[2 * i + 1] += v * w[2 * r + 1]; } }
    y
}
pub fn sur(step: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], ys: &[Vec<f64>], k: usize) -> f64 {
    let mut z = x0.to_vec(); let mut r = 0.0;
    for t in 0..k { z = step(&z); r += if fin(&z) { m::rel_err(&z, &ys[t]).min(10.0) } else { 10.0 }; }
    r / k as f64
}
/// Gauss-Newton/Levenberg-Marquardt sobre la pérdida multipaso desde x0: Σ_t ||(F^t(x0) − y_t)/||y_t|| ||² + λ||θ − θ_prior||²
pub fn fit_gn(step: &dyn Fn(&[f64], &[f64]) -> Vec<f64>, prior: &[f64], lam: f64, x0: &[f64], ys: &[Vec<f64>], k: usize) -> Vec<f64> {
    let res = |th: &[f64]| -> Option<Vec<f64>> {
        let mut z = x0.to_vec(); let mut r = vec![];
        for t in 0..k { z = step(th, &z); if !fin(&z) || z.iter().any(|v| v.abs() > 1e6) { return None; } let n = m::norm(&ys[t]).max(1e-9); for (a, b) in z.iter().zip(&ys[t]) { r.push((a - b) / n); } }
        Some(r)
    };
    let obj = |th: &[f64], r: &[f64]| r.iter().map(|v| v * v).sum::<f64>() + lam * th.iter().zip(prior).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
    let mut th = prior.to_vec(); let Some(mut r) = res(&th) else { return th; }; let mut f = obj(&th, &r); let np = th.len(); let mut mu = 1e-2;
    for _ in 0..12 {
        let nr = r.len(); let mut jt = vec![0.0; np * nr];
        for p in 0..np { let mut t2 = th.clone(); let h = 1e-6 * (1.0 + th[p].abs()); t2[p] += h; if let Some(r2) = res(&t2) { for i in 0..nr { jt[p * nr + i] = (r2[i] - r[i]) / h; } } }
        let mut hm = vec![0.0; np * np]; let mut g = vec![0.0; np];
        for p in 0..np { for q in 0..=p { let s: f64 = (0..nr).map(|i| jt[p * nr + i] * jt[q * nr + i]).sum(); hm[p * np + q] = s; hm[q * np + p] = s; }
            g[p] = (0..nr).map(|i| jt[p * nr + i] * r[i]).sum::<f64>() + lam * (th[p] - prior[p]); }
        let mut ok = false; let f0 = f;
        for _ in 0..6 {
            let mut h2 = hm.clone(); for p in 0..np { h2[p * np + p] += lam + mu; }
            let dl = solve(&h2, &g, np); let cand: Vec<f64> = th.iter().zip(&dl).map(|(a, b)| a - b).collect();
            if let Some(rc) = res(&cand) { let fc = obj(&cand, &rc); if fc < f { th = cand; r = rc; f = fc; mu = (mu / 3.0).max(1e-9); ok = true; break; } }
            mu *= 5.0;
        }
        if !ok || f0 - f < 1e-10 * (1.0 + f0) { break; }
    }
    th
}
#[derive(Clone)]
pub struct Entry { pub g0: usize, pub a: Vec<f64>, pub fam: usize }
pub struct Ep { pub idx: usize, pub tid: String, pub fam: usize, pub ph: [f64; 2], pub c: Vec<f64>, pub x0: Vec<f64>, pub ys: Vec<Vec<f64>> }
/// decisión LAT (memoria): None = conocido (OBS). Devuelve (g0, A) a usar, acción, pureza del prior (peso en entradas de la misma familia verdadera)
pub fn lat_decide(d: &Orbita, cx: &Ctx8, lib: &[Entry], e: &Ep, k: usize, sk: f64, gs: usize, tau: f64, lam0: f64, lamm: f64) -> (Option<(usize, Vec<f64>)>, &'static str, f64) {
    let mut sl: Vec<(f64, usize)> = lib.iter().enumerate().map(|(i, en)| (sur(&|z| lat_step(d, &cx.cps[en.g0], &cx.cs[en.g0], &en.a, z), &e.x0, &e.ys, k), i)).collect();
    sl.sort_by(|a, b| a.0.total_cmp(&b.0));
    let slib = sl.first().map_or(f64::INFINITY, |v| v.0);
    if sk <= tau && sk <= slib { return (None, "known", f64::NAN); }
    if slib < sk {
        let g0 = lib[sl[0].1].g0; let mut prior = vec![0.0; npar()]; let (mut ws, mut wsame) = (0.0, 0.0);
        for (s, i) in sl.iter().filter(|(_, i)| lib[*i].g0 == g0).take(TOPM) { let w = (-(s - slib) / tau).exp(); ws += w; if lib[*i].fam == e.fam { wsame += w; } for (p, v) in prior.iter_mut().zip(&lib[*i].a) { *p += w * v; } }
        for p in prior.iter_mut() { *p /= ws; }
        let a = fit_gn(&|th, z| lat_step(d, &cx.cps[g0], &cx.cs[g0], th, z), &prior, lamm, &e.x0, &e.ys, k);
        return (Some((g0, a)), "lib", wsame / ws);
    }
    let a = fit_gn(&|th, z| lat_step(d, &cx.cps[gs], &cx.cs[gs], th, z), &vec![0.0; npar()], lam0, &e.x0, &e.ys, k);
    (Some((gs, a)), "new", f64::NAN)
}
pub fn ok16(e: &Ep, step: &dyn Fn(&[f64]) -> Vec<f64>, k: usize) -> u8 { let mut z = e.ys[k - 1].clone(); for _ in k + 1..=16 { z = step(&z); } (fin(&z) && m::correct(&z, &e.ys[15])) as u8 }
pub fn load_set(p: &str) -> Vec<(usize, usize, [f64; 2], Vec<f64>)> {
    std::fs::read_to_string(p).unwrap().lines().skip(1).map(|l| { let f: Vec<&str> = l.split(',').collect(); (f[0].parse().unwrap(), f[1].parse().unwrap(), [f[2].parse().unwrap(), f[3].parse().unwrap()], nums(f[4], ';')) }).collect()
}
pub fn load_eps(ctx: &str, set: &[(usize, usize, [f64; 2], Vec<f64>)]) -> Vec<(Ep, String)> {
    let mut by = std::collections::HashMap::new(); for s in set { by.insert(s.0, s); }
    std::fs::read_to_string(ctx).unwrap().lines().skip(1).map(|line| {
        let f: Vec<&str> = line.split(',').collect(); let idx: usize = f[0].parse().unwrap(); let c: Vec<f64> = f[3..13].iter().map(|v| v.parse().unwrap()).collect();
        let (_, fam, pt, x0) = by[&idx]; let mut ys = vec![]; let mut y = x0.clone(); for _ in 0..16 { y = rule(*fam, *pt, &y); ys.push(y.clone()); }
        (Ep { idx, tid: f[2].to_string(), fam: *fam, ph: [c[8], c[9]], c, x0: x0.clone(), ys }, f[1].to_string())
    }).collect()
}
/// reid + OBS + ORACLE (por episodio) comunes
pub fn common(d: &Orbita, e: &Ep, k: usize) -> (f64, usize, [f64; 2], u8) {
    let (g, pp, r) = reid(d, &e.x0, &e.ys, k, e.ph); let cg = context_of(g, pp); (r / k as f64, g, pp, ok16(e, &|z| d.step(z, &cg), k))
}
pub fn oep16(d: &Orbita, e: &Ep, k: usize, deg: usize, lam: f64) -> u8 { let mut o = Res::new(None, deg, lam); o.add(d, &e.x0, &e.ys, k, e.ph, e.fam); ok16(e, &|z| o.step(d, z, e.ph), k) }
pub fn oms16(e: &Ep, k: usize, deg: usize, lam: f64) -> u8 { let n = 2 * mono(0.0, 0.0, 0, deg).len(); let w = fit_gn(&|th, z| dir_step(th, deg, z), &vec![0.0; n], lam, &e.x0, &e.ys, k); ok16(e, &|z| dir_step(&w, deg, z), k) }
fn argf(a: &[String], k: &str) -> f64 { super::arg(a, k).unwrap().parse().unwrap() }

pub fn run(a: &[String]) {
    let wtxt = std::fs::read_to_string(super::arg(a, "--weights").unwrap()).unwrap(); let d = Orbita::from_json(&wtxt);
    let set = load_set(&super::arg(a, "--set").unwrap()); let before = cdt_rqm_epr::v4_provenance::snapshot();
    let (lam0, lamm) = (argf(a, "--lam0"), argf(a, "--lamm"));
    if a[2] == "exp1" {
        let k: usize = super::arg(a, "--k").unwrap().parse().unwrap(); let tau = argf(a, "--tau");
        let eps = load_eps(&super::arg(a, "--ctx").unwrap(), &set);
        let mut hdr = "pos,idx,text_id,fam,k,s_known,g_star,base16,obs16,res16,resep16,latep16,latmem16,latmem_action,latmem_purity,ls16".to_string();
        for (dg, l) in OEP_GRID { hdr += &format!(",oep_d{dg}_l{l}"); } for (dg, l) in OMS_GRID { hdr += &format!(",oms_d{dg}_l{l}"); } hdr += ",final16,final_action";
        println!("{hdr}");
        let mut lib: Vec<Entry> = vec![]; let mut rlib: Vec<Res> = vec![]; let mut rows = vec![]; let mut keep = vec![];
        for (pos, (e, _)) in eps.iter().enumerate() {
            let (sk, gs, pp, obs) = common(&d, e, k); let cx = ctx8(&d, e.ph);
            let base = { let mut z = e.x0.clone(); for _ in 0..16 { z = d.step(&z, &e.c); } (fin(&z) && m::correct(&z, &e.ys[15])) as u8 };
            // RES (e2ac4ab, biblioteca antigua) y RES_EP
            let sl: Vec<f64> = rlib.iter().map(|mm| mm.surprise(&d, &e.x0, &e.ys, k, e.ph)).collect();
            let (mut best, mut pick) = (sk, -1i64); for (i, s) in sl.iter().enumerate() { if *s < best { best = *s; pick = i as i64; } }
            let mid: i64 = if best <= tau { if pick >= 0 { rlib[pick as usize].add(&d, &e.x0, &e.ys, k, e.ph, e.fam); } pick } else {
                let mut jb = (f64::INFINITY, usize::MAX, None);
                for (i, mm) in rlib.iter().enumerate() { let mut m2 = mm.clone(); m2.add(&d, &e.x0, &e.ys, k, e.ph, e.fam); let s = m2.surprise(&d, &e.x0, &e.ys, k, e.ph); if s < jb.0 { jb = (s, i, Some(m2)); } }
                if jb.0 <= tau { rlib[jb.1] = jb.2.unwrap(); jb.1 as i64 } else { let mut mm = Res::new(Some(gs), RES_DEG, RES_LAM); mm.add(&d, &e.x0, &e.ys, k, e.ph, e.fam); rlib.push(mm); (rlib.len() - 1) as i64 } };
            let res = if mid < 0 { obs } else { let mm = &rlib[mid as usize]; ok16(e, &|z| mm.step(&d, z, e.ph), k) };
            let resep = if sk <= tau { obs } else { let mut mm = Res::new(Some(gs), RES_DEG, RES_LAM); mm.add(&d, &e.x0, &e.ys, k, e.ph, e.fam); ok16(e, &|z| mm.step(&d, z, e.ph), k) };
            // LAT_EP y LAT (memoria)
            let latep = if sk <= tau { obs } else { let aa = fit_gn(&|th, z| lat_step(&d, &cx.cps[gs], &cx.cs[gs], th, z), &vec![0.0; npar()], lam0, &e.x0, &e.ys, k); ok16(e, &|z| lat_step(&d, &cx.cps[gs], &cx.cs[gs], &aa, z), k) };
            let (dec, act, pur) = lat_decide(&d, &cx, &lib, e, k, sk, gs, tau, lam0, lamm);
            let lat = match &dec { None => obs, Some((g0, aa)) => ok16(e, &|z| lat_step(&d, &cx.cps[*g0], &cx.cs[*g0], aa, z), k) };
            if let Some((g0, aa)) = dec { lib.push(Entry { g0, a: aa, fam: e.fam }); }
            let mut pr: Vec<(&[f64], &[f64])> = vec![(&e.x0, &e.ys[0])]; for t in 1..k { pr.push((&e.ys[t - 1], &e.ys[t])); }
            let wl = ls_fit(&pr); let ls = ok16(e, &|z| ls_step(&wl, z), k);
            let mut row = format!("{pos},{},{},{},{k},{sk:.8},{gs},{base},{obs},{res},{resep},{latep},{lat},{act},{pur:.4},{ls}", e.idx, e.tid, e.fam);
            for (dg, l) in OEP_GRID { row += &format!(",{}", oep16(&d, e, k, dg, l)); } for (dg, l) in OMS_GRID { row += &format!(",{}", oms16(e, k, dg, l)); }
            rows.push(row); keep.push((sk, gs, pp, obs));
        }
        // FINAL: memoria congelada (sin almacenar)
        for (i, (e, _)) in eps.iter().enumerate() {
            let (sk, gs, _, obs) = keep[i]; let cx = ctx8(&d, e.ph);
            let (dec, act, _) = lat_decide(&d, &cx, &lib, e, k, sk, gs, tau, lam0, lamm);
            let fv = match &dec { None => obs, Some((g0, aa)) => ok16(e, &|z| lat_step(&d, &cx.cps[*g0], &cx.cs[*g0], aa, z), k) };
            println!("{},{fv},{act}", rows[i]);
        }
        eprintln!("queries={} lib_size={} res_modules={}", cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(), lib.len(), rlib.len());
    } else {
        // flujo largo
        let taus = nums(&super::arg(a, "--taus").unwrap(), ','); let tau_of = |k: usize| match k { 2 => taus[0], 4 => taus[1], _ => taus[2] };
        let stream = load_eps(&super::arg(a, "--ctx").unwrap(), &set); let hset = load_set(&super::arg(a, "--hset").unwrap());
        let held = load_eps(&super::arg(a, "--held").unwrap(), &hset);
        let (od, ol, md, ml) = (argf(a, "--oep-d") as usize, argf(a, "--oep-l"), argf(a, "--oms-d") as usize, argf(a, "--oms-l"));
        let ckpts: Vec<usize> = nums(&super::arg(a, "--ckpts").unwrap(), ',').iter().map(|v| *v as usize).collect();
        let mut hout = String::from("ckpt,idx,text_id,fam,k,obs16,oep16,oms16,lat16,action,purity,lib_size\n");
        // precálculo held-out por (episodio, k): independiente del tiempo
        let hk: Vec<Vec<(f64, usize, [f64; 2], u8, u8, u8)>> = held.iter().map(|(e, _)| [2usize, 4, 8].iter().map(|&k| { let (sk, gs, pp, obs) = common(&d, e, k); (sk, gs, pp, obs, oep16(&d, e, k, od, ol), oms16(e, k, md, ml)) }).collect()).collect();
        let mut lib: Vec<Entry> = vec![];
        let evalh = |lib: &[Entry], n: usize, hout: &mut String| {
            for (j, (e, _)) in held.iter().enumerate() { let cx = ctx8(&d, e.ph);
                for (ki, &k) in [2usize, 4, 8].iter().enumerate() { let (sk, gs, _, obs, oe, om) = hk[j][ki];
                    let (dec, act, pur) = lat_decide(&d, &cx, lib, e, k, sk, gs, tau_of(k), lam0, lamm);
                    let lv = match &dec { None => obs, Some((g0, aa)) => ok16(e, &|z| lat_step(&d, &cx.cps[*g0], &cx.cs[*g0], aa, z), k) };
                    hout.push_str(&format!("{n},{},{},{},{k},{obs},{oe},{om},{lv},{act},{pur:.4},{}\n", e.idx, e.tid, e.fam, lib.len())); } }
        };
        println!("pos,idx,text_id,fam,k,s_known,obs16,latep16,lat16,action,purity,lib_size");
        for (pos, (e, cond)) in stream.iter().enumerate() {
            if ckpts.contains(&pos) { evalh(&lib, pos, &mut hout); }
            let k: usize = cond.trim_start_matches('k').parse().unwrap(); let tau = tau_of(k);
            let (sk, gs, _, obs) = common(&d, e, k); let cx = ctx8(&d, e.ph);
            let latep = if sk <= tau { obs } else { let aa = fit_gn(&|th, z| lat_step(&d, &cx.cps[gs], &cx.cs[gs], th, z), &vec![0.0; npar()], lam0, &e.x0, &e.ys, k); ok16(e, &|z| lat_step(&d, &cx.cps[gs], &cx.cs[gs], &aa, z), k) };
            let (dec, act, pur) = lat_decide(&d, &cx, &lib, e, k, sk, gs, tau, lam0, lamm);
            let lv = match &dec { None => obs, Some((g0, aa)) => ok16(e, &|z| lat_step(&d, &cx.cps[*g0], &cx.cs[*g0], aa, z), k) };
            if let Some((g0, aa)) = dec { lib.push(Entry { g0, a: aa, fam: e.fam }); }
            println!("{pos},{},{},{},{k},{sk:.8},{obs},{latep},{lv},{act},{pur:.4},{}", e.idx, e.tid, e.fam, lib.len());
        }
        if ckpts.contains(&stream.len()) { evalh(&lib, stream.len(), &mut hout); }
        std::fs::write(super::arg(a, "--held-out").unwrap(), hout).unwrap();
        eprintln!("queries={} lib_size={}", cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(), lib.len());
    }
    assert_eq!(Orbita::from_json(&wtxt).raw, d.raw);
}
