//! ÓRBITA-Nova-2 · Paso 0 — C6 arreglado, SOLO SMOKE (docs/orbita_nova2/PASO0_C6.md).
//!   nova2-p0-calib --wdir DIR --out DIR
//!   nova2-p0 --weights W --seed S --tau T --out DIR [--n-stream N --h-per-fam H --consolidate-every E]
//! Variante opt-in: no toca nova/nova_train/nova2*. Núcleo ÓRBITA congelado.
#![allow(clippy::too_many_arguments)]
use crate::latent::fit_gn;
use crate::model::{Orbita, M};
use crate::nova::{arbitrate, gate_nodamage, lr_step, OperatorLib, DEFAULT_RANK};
use crate::nova2::err_c3;
use crate::nova_train::{
    consolidate, ctx_for_eval, fit_c4, gen_calib_h, gen_heldout_n, gen_stream, k_fit_val, log_rel, lora_prior, med_f64,
    BufItem, Ep, EPS_SPEC, LAM_C4, PRIMARY_NEW,
};
use cdt_rqm_epr::v4_dataset::N_FAM;
use std::io::Write;
use std::path::PathBuf;

pub const LAM_PRIOR: f64 = 1e-2;
/// τ legado (Nova f64b289) sólo para la biblioteca sombra.
const TAU_LEGACY: f64 = 0.9386641928884383;

/// vec(UVᵀ), fila-mayor M×M.
pub fn dk_of(th: &[f64], r: usize) -> Vec<f64> {
    let n = r * M;
    let (u, v) = (&th[..n], &th[n..]);
    let mut k = vec![0.0; M * M];
    for s in 0..r {
        for i in 0..M {
            for j in 0..M {
                k[i * M + j] += u[s * M + i] * v[s * M + j];
            }
        }
    }
    k
}

/// SVD truncada rango r por potencia + deflación → θ=(U,V) con U=u√σ, V=v√σ.
pub fn th_of_dk(dk: &[f64], r: usize) -> Vec<f64> {
    let mut a = dk.to_vec();
    let mut th = vec![0.0; 2 * r * M];
    for s in 0..r {
        let mut v = vec![0.0; M];
        for (j, x) in v.iter_mut().enumerate() {
            *x = 1.0 + 0.1 * j as f64;
        }
        let mut sig = 0.0;
        let mut u = vec![0.0; M];
        for _ in 0..200 {
            for i in 0..M {
                u[i] = (0..M).map(|j| a[i * M + j] * v[j]).sum();
            }
            let nu = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nu < 1e-300 {
                break;
            }
            u.iter_mut().for_each(|x| *x /= nu);
            for j in 0..M {
                v[j] = (0..M).map(|i| a[i * M + j] * u[i]).sum();
            }
            sig = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if sig < 1e-300 {
                break;
            }
            v.iter_mut().for_each(|x| *x /= sig);
        }
        if sig < 1e-300 {
            break;
        }
        let rs = sig.sqrt();
        for i in 0..M {
            th[s * M + i] = u[i] * rs;
            th[r * M + s * M + i] = v[i] * rs;
        }
        for i in 0..M {
            for j in 0..M {
                a[i * M + j] -= sig * u[i] * v[j];
            }
        }
    }
    th
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

#[derive(Clone)]
pub struct P0Entry { pub sig: Vec<f64>, pub dk: Vec<f64>, pub th: Vec<f64>, pub n: usize, pub fam_n: Vec<usize> }
#[derive(Clone, Default)]
pub struct P0Lib { pub entries: Vec<P0Entry>, pub merges: usize, pub adds: usize }
impl P0Lib {
    pub fn retrieve(&self, sig: &[f64], tau: f64) -> Option<&P0Entry> {
        let mut best: Option<(f64, &P0Entry)> = None;
        for e in &self.entries {
            let d = dist(&e.sig, sig);
            if d <= tau && best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, e));
            }
        }
        best.map(|x| x.1)
    }
    pub fn shuffled(&self, seed: u64) -> P0Lib {
        use rand::seq::SliceRandom;
        use rand_xoshiro::rand_core::SeedableRng;
        let mut out = self.clone();
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed ^ 0x5A1F_51C6u64);
        let mut s: Vec<Vec<f64>> = out.entries.iter().map(|e| e.sig.clone()).collect();
        s.shuffle(&mut rng);
        for (e, x) in out.entries.iter_mut().zip(s) {
            e.sig = x;
        }
        out
    }
}

pub struct P0Item { pub sig: Vec<f64>, pub dk: Vec<f64>, pub fam: usize, pub known: bool, pub val: f64, pub base: f64 }

/// Clustering greedy en el buffer (firma ΔK, τ) → media ΔK → fusión con la entrada más cercana (≤τ) o alta.
pub fn consolidate_p0(lib: &mut P0Lib, buf: &mut Vec<P0Item>, tau: f64, rank: usize) {
    let mut cl: Vec<Vec<usize>> = vec![];
    let mut ce: Vec<Vec<f64>> = vec![];
    for (i, b) in buf.iter().enumerate() {
        let best = ce.iter().enumerate().map(|(c, x)| (c, dist(x, &b.sig))).filter(|x| x.1 <= tau).min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((c, _)) = best {
            cl[c].push(i);
            let n = cl[c].len() as f64;
            for (a, v) in ce[c].iter_mut().zip(&b.sig) {
                *a = (*a * (n - 1.0) + v) / n;
            }
        } else {
            cl.push(vec![i]);
            ce.push(b.sig.clone());
        }
    }
    for (c, mem) in cl.iter().enumerate() {
        let n = mem.len();
        let damaged = mem.iter().any(|&i| buf[i].known && buf[i].val > buf[i].base + 1e-12);
        let mv = mem.iter().map(|&i| buf[i].val).sum::<f64>() / n as f64;
        let mb = mem.iter().map(|&i| buf[i].base).sum::<f64>() / n as f64;
        if damaged || !(mv < mb - 1e-12) {
            continue;
        }
        let mut dk = vec![0.0; M * M];
        let mut fam_n = vec![0usize; 16];
        for &i in mem {
            for (a, v) in dk.iter_mut().zip(&buf[i].dk) {
                *a += v / n as f64;
            }
            fam_n[buf[i].fam.min(15)] += 1;
        }
        let best = lib.entries.iter().enumerate().map(|(j, e)| (j, dist(&e.sig, &ce[c]))).filter(|x| x.1 <= tau).min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((j, _)) = best {
            let e = &mut lib.entries[j];
            let (w0, w1) = (e.n as f64, n as f64);
            for (a, v) in e.dk.iter_mut().zip(&dk) {
                *a = (*a * w0 + v * w1) / (w0 + w1);
            }
            e.sig = e.dk.clone();
            e.th = th_of_dk(&e.dk, rank);
            e.n += n;
            for (a, v) in e.fam_n.iter_mut().zip(&fam_n) {
                *a += v;
            }
            lib.merges += 1;
        } else {
            let th = th_of_dk(&dk, rank);
            lib.entries.push(P0Entry { sig: dk.clone(), dk, th, n, fam_n });
            lib.adds += 1;
        }
    }
    buf.clear();
}

fn dom_fam(e: &P0Entry) -> usize {
    e.fam_n.iter().enumerate().max_by_key(|x| x.1).map(|x| x.0).unwrap_or(0)
}

/// NOVA con C6 arreglado. lib=None → NOVA−C6. Devuelve (θ, aceptado, hit, puro).
fn nova_fit(d: &Orbita, e: &Ep, cg: &[f64], lib: Option<&P0Lib>, tau: f64, rank: usize, prior0: &[f64], k: usize) -> (Vec<f64>, bool, bool, bool, Vec<f64>) {
    let cp = d.ctx(cg);
    let (kf, kv) = k_fit_val(k);
    let mut ec = e.clone();
    ec.c = cg.to_vec();
    let th0 = fit_c4(d, &ec, prior0, rank, k);
    let sig = dk_of(&th0, rank);
    let hit = lib.and_then(|l| l.retrieve(&sig, tau));
    let (mu, lam) = match hit { Some(en) => (en.th.clone(), LAM_PRIOR), None => (prior0.to_vec(), LAM_C4) };
    let pure = hit.map_or(false, |en| dom_fam(en) == e.fam);
    let th = fit_gn(&|t, z| lr_step(d, &cp, cg, t, rank, z), &mu, lam, &e.x0, &e.ys, k);
    let ver = arbitrate(d, &cp, cg, &th, rank, &e.x0, &e.ys, kf, kv, EPS_SPEC);
    let acc = ver.accept && gate_nodamage(ver.val_err, ver.base_val);
    let mut vb = vec![ver.val_err, ver.base_val];
    vb.extend(sig);
    (th, acc, hit.is_some(), pure, vb)
}

const ARMS: [&str; 4] = ["orbita_c3", "nova", "nova_m_c6", "nova_shuf"];

fn aggregate(d: &Orbita, held: &[Ep], lib: &P0Lib, tau: f64, rank: usize, seed: u64, k: usize) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let mut per: Vec<Vec<f64>> = vec![vec![]; 4];
    let (mut hits, mut pure, mut hits_s, mut thdiff, mut nh) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut cols: Vec<(Vec<f64>, Vec<f64>, usize, usize, usize, usize)> = vec![(vec![], vec![], 0, 0, 0, 0); 4];
    for e in held {
        let (cg, _, _) = ctx_for_eval(d, e, k);
        let cp = d.ctx(&cg);
        let prior0 = lora_prior(seed, e.fam as u64, rank);
        let shuf = lib.shuffled(seed ^ (e.fam as u64) << 16);
        let mut ths: Vec<Vec<f64>> = vec![];
        for (ai, name) in ARMS.iter().enumerate() {
            let (err, ok) = if *name == "orbita_c3" {
                err_c3(d, &cg, e, k, &|z| d.step(z, &cg))
            } else {
                let l = match *name { "nova" => Some(lib), "nova_shuf" => Some(&shuf), _ => None };
                let (th, acc, h, p, _) = nova_fit(d, e, &cg, l, tau, rank, &prior0, k);
                if *name == "nova" { hits += h as usize; pure += p as usize; }
                if *name == "nova_shuf" { hits_s += h as usize; }
                ths.push(th.clone());
                let ada = |z: &[f64]| lr_step(d, &cp, &cg, &th, rank, z);
                let base = |z: &[f64]| d.step(z, &cg);
                let st: &dyn Fn(&[f64]) -> Vec<f64> = if acc { &ada } else { &base };
                err_c3(d, &cg, e, k, st)
            };
            per[ai].push(err);
            let c = &mut cols[ai];
            if PRIMARY_NEW.contains(&e.fam) { c.0.push(log_rel(err)); c.2 += 1; c.4 += ok as usize; }
            else if e.fam < N_FAM { c.1.push(log_rel(err)); c.3 += 1; c.5 += ok as usize; }
        }
        nh += 1;
        if dist(&ths[0], &ths[1]) > 1e-12 { thdiff += 1; }
    }
    for (ai, name) in ARMS.iter().enumerate() {
        let c = &mut cols[ai];
        out.insert((*name).into(), serde_json::json!({
            "new_med_log": med_f64(&mut c.0), "known_med_log": med_f64(&mut c.1),
            "new_acc16": c.4 as f64 / c.2.max(1) as f64, "known_acc16": c.5 as f64 / c.3.max(1) as f64, "n_new": c.2, "n_known": c.3,
        }));
    }
    let nd = |a: usize, b: usize| per[a].iter().zip(&per[b]).filter(|(x, y)| (*x - *y).abs() > 1e-12).count();
    out.insert("diag".into(), serde_json::json!({
        "n_held": nh, "nova_hits": hits, "nova_hit_pure": pure, "shuf_hits": hits_s, "theta_diff_nova_vs_mc6": thdiff,
        "err_diff_nova_vs_mc6": nd(1, 2), "err_diff_nova_vs_shuf": nd(1, 3),
    }));
    serde_json::Value::Object(out)
}

fn arg_of(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

fn pct(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let x = p / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (x.floor() as usize, x.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (x - lo as f64)
}

pub fn run_calib(a: &[String]) {
    let wdir = PathBuf::from(arg_of(a, "--wdir").expect("--wdir"));
    let out = PathBuf::from(arg_of(a, "--out").expect("--out"));
    std::fs::create_dir_all(&out).unwrap();
    let rank = DEFAULT_RANK;
    let (mut dk_d, mut th_d) = (vec![], vec![]);
    for i in 0..16u64 {
        let seed = 0xA400 + i;
        let d = Orbita::from_json(&std::fs::read_to_string(wdir.join(format!("weights_0x{seed:X}.json"))).unwrap());
        let eps = gen_calib_h(seed, 16);
        let it: Vec<(usize, Vec<f64>, Vec<f64>)> = eps.iter().enumerate().map(|(j, e)| {
            let th = fit_c4(&d, e, &lora_prior(seed, j as u64, rank), rank, 4);
            (e.fam, dk_of(&th, rank), th)
        }).collect();
        for (fi, si, ti) in &it {
            let o = it.iter().filter(|x| x.0 != *fi);
            dk_d.push(o.clone().map(|x| dist(si, &x.1)).fold(f64::INFINITY, f64::min));
            th_d.push(o.map(|x| dist(ti, &x.2)).fold(f64::INFINITY, f64::min));
        }
    }
    let tau = pct(dk_d.clone(), 5.0);
    let rep = serde_json::json!({
        "tau": tau, "rule": "P5 distancia L2 firma vec(ΔK) al vecino más cercano de la otra familia; H DEV fam 1–2; 16 seeds DEV; k=4",
        "n": dk_d.len(), "dk_p5": tau, "dk_p50": pct(dk_d.clone(), 50.0), "dk_p95": pct(dk_d, 95.0),
        "th_p5": pct(th_d.clone(), 5.0), "th_p50": pct(th_d.clone(), 50.0), "th_p95": pct(th_d, 95.0),
        "doc": "docs/orbita_nova2/PASO0_C6.md",
    });
    std::fs::write(out.join("tau.json"), serde_json::to_string_pretty(&rep).unwrap()).unwrap();
    eprintln!("p0 tau={tau:.6e} {rep}");
}

pub fn run(a: &[String]) {
    let wtxt = std::fs::read_to_string(arg_of(a, "--weights").expect("--weights")).unwrap();
    let d = Orbita::from_json(&wtxt);
    let before = d.raw.clone();
    let s = arg_of(a, "--seed").expect("--seed");
    let seed = u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap();
    assert!((0xA400..=0xA40F).contains(&seed), "Paso 0: sólo semillas DEV 0xA400..0xA40F");
    let tau: f64 = arg_of(a, "--tau").expect("--tau").parse().unwrap();
    let n_stream: usize = arg_of(a, "--n-stream").map(|s| s.parse().unwrap()).unwrap_or(2400);
    let hpf: usize = arg_of(a, "--h-per-fam").map(|s| s.parse().unwrap()).unwrap_or(4);
    let every: usize = arg_of(a, "--consolidate-every").map(|s| s.parse().unwrap()).unwrap_or(50);
    let out_dir = PathBuf::from(arg_of(a, "--out").expect("--out"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let rank = DEFAULT_RANK;
    let stream = gen_stream(seed, n_stream);
    let held = gen_heldout_n(seed, hpf);
    let mut lib = P0Lib::default();
    let mut buf: Vec<P0Item> = vec![];
    let mut shadow = OperatorLib::new();
    let mut sbuf: Vec<BufItem> = vec![];
    let (mut acc, mut hits, mut pure) = (0usize, 0usize, 0usize);
    let mut rows = vec![];
    let ck: Vec<usize> = [0usize, 600, 1200, n_stream].into_iter().filter(|&c| c <= n_stream).collect();
    let t0 = std::time::Instant::now();
    let mut evalc = |n: usize, lib: &P0Lib, sh: &OperatorLib, acc: usize, hits: usize, pure: usize| {
        let r = serde_json::json!({
            "n": n, "lib_size": lib.entries.len(), "lib_merges": lib.merges, "lib_adds": lib.adds,
            "lib_support": lib.entries.iter().map(|e| e.n).collect::<Vec<_>>(),
            "lib_dom_fam": lib.entries.iter().map(dom_fam).collect::<Vec<_>>(),
            "shadow_legacy_lib_size": sh.entries.len(),
            "stream_accept": acc, "stream_hits": hits, "stream_hit_pure": pure,
            "k2": aggregate(&d, &held, lib, tau, rank, seed, 2),
            "k4": aggregate(&d, &held, lib, tau, rank, seed, 4),
            "k8": aggregate(&d, &held, lib, tau, rank, seed, 8),
        });
        eprintln!("0x{seed:X} n={n} lib={} shadow={} hits={hits} pure={pure} t={:.0}s", lib.entries.len(), sh.entries.len(), t0.elapsed().as_secs_f64());
        rows.push(r);
    };
    if ck.contains(&0) { evalc(0, &lib, &shadow, 0, 0, 0); }
    for (pos, e) in stream.iter().enumerate() {
        let k = e.k;
        let (cg, _, _) = ctx_for_eval(&d, e, k);
        let prior0 = lora_prior(seed, (pos as u64) << 8 ^ e.fam as u64, rank);
        let (th, ok, h, p, vb) = nova_fit(&d, e, &cg, Some(&lib), tau, rank, &prior0, k);
        hits += h as usize;
        pure += p as usize;
        if ok {
            acc += 1;
            let sig = vb[2..].to_vec();
            let known = e.fam < N_FAM;
            buf.push(P0Item { sig, dk: dk_of(&th, rank), fam: e.fam, known, val: vb[0], base: vb[1] });
            sbuf.push(BufItem { signature: th.clone(), th: th.clone(), _fam: e.fam, known, val_err: vb[0], base_err: vb[1] });
        }
        let n = pos + 1;
        if (every > 0 && n % every == 0) || ck.contains(&n) {
            consolidate_p0(&mut lib, &mut buf, tau, rank);
            if !sbuf.is_empty() { consolidate(&mut shadow, &mut sbuf, TAU_LEGACY); }
        }
        if ck.contains(&n) { evalc(n, &lib, &shadow, acc, hits, pure); }
    }
    let rep = serde_json::json!({
        "seed": format!("0x{seed:X}"), "phase": "smoke_dev", "n_stream": n_stream, "h_per_fam": hpf, "every": every,
        "tau": tau, "lam_prior": LAM_PRIOR, "rank": rank, "ckpts": rows, "arms": ARMS, "doc": "docs/orbita_nova2/PASO0_C6.md",
    });
    let mut f = std::fs::File::create(out_dir.join(format!("metrics_0x{seed:X}.json"))).unwrap();
    writeln!(f, "{}", serde_json::to_string_pretty(&rep).unwrap()).unwrap();
    assert_eq!(Orbita::from_json(&wtxt).raw, before, "núcleo ÓRBITA modificado");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dk_roundtrip_and_invariance() {
        let r = DEFAULT_RANK;
        let th: Vec<f64> = (0..2 * r * M).map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.01).collect();
        let k = dk_of(&th, r);
        let k2 = dk_of(&th_of_dk(&k, r), r);
        assert!(dist(&k, &k2) < 1e-9 * (1.0 + k.iter().map(|x| x.abs()).sum::<f64>()));
        // U→2U, V→V/2 deja ΔK igual (firma identificable)
        let mut t2 = th.clone();
        for i in 0..r * M { t2[i] *= 2.0; t2[r * M + i] /= 2.0; }
        assert!(dist(&dk_of(&t2, r), &k) < 1e-12);
    }
}
