//! Aprendizaje online de dinámicas (docs/prereg_orbita_aprendizaje_online.md). No lee ds.test; no modifica ÓRBITA (núcleo congelado).
//! learn-eval --weights W --set S --ctx C --k K --deg D --lam L --tau T
//!   C: filas en orden de flujo (idx,cond,text_id,c0..c9). Salida: CSV por episodio (todas las condiciones) + FINAL; stderr: queries + biblioteca JSON.
use crate::fam::{ls_fit, ls_step, reid, rule};
use crate::model::Orbita;
use cdt_rqm_epr::v4_dataset::context_of;
use cdt_rqm_epr::v4_metrics as m;

fn fin(z: &[f64]) -> bool { z.iter().all(|v| v.is_finite()) }
fn nums(s: &str, sep: char) -> Vec<f64> { s.split(sep).filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect() }

/// Φ(q, p) = monomios x^i y^j (i + j ≤ deg) ⊗ [1, p1, p2]
fn phi(x: f64, y: f64, p: [f64; 2], deg: usize) -> Vec<f64> {
    let mut mono = vec![];
    for t in 0..=deg { for j in 0..=t { mono.push(x.powi((t - j) as i32) * y.powi(j as i32)); } }
    let mut out = Vec::with_capacity(mono.len() * 3);
    for s in [1.0, p[0], p[1]] { for v in &mono { out.push(v * s); } }
    out
}
fn nfeat(deg: usize) -> usize { 3 * (deg + 1) * (deg + 2) / 2 }
/// Cholesky de A (n×n, SPD) y resolución A w = b
fn solve(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n { for j in 0..=i {
        let mut s = a[i * n + j]; for k in 0..j { s -= l[i * n + k] * l[j * n + k]; }
        if i == j { l[i * n + i] = s.max(1e-300).sqrt(); } else { l[i * n + j] = s / l[j * n + j]; }
    } }
    let mut z = vec![0.0; n];
    for i in 0..n { let mut s = b[i]; for k in 0..i { s -= l[i * n + k] * z[k]; } z[i] = s / l[i * n + i]; }
    let mut w = vec![0.0; n];
    for i in (0..n).rev() { let mut s = z[i]; for k in i + 1..n { s -= l[k * n + i] * w[k]; } w[i] = s / l[i * n + i]; }
    w
}

/// Módulo de dinámica aprendido: F(q) = base(q) + Wᵀ Φ(q, p̂); base = ÓRBITA(·, c(g0, p̂)) congelada, o identidad (oráculo directo).
#[derive(Clone)]
pub struct Module { pub g0: Option<usize>, deg: usize, lam: f64, a: Vec<f64>, b: [Vec<f64>; 2], w: [Vec<f64>; 2], pub members: Vec<usize>, pub npairs: usize }
impl Module {
    fn new(g0: Option<usize>, deg: usize, lam: f64) -> Self {
        let n = nfeat(deg);
        Self { g0, deg, lam, a: vec![0.0; n * n], b: [vec![0.0; n], vec![0.0; n]], w: [vec![0.0; n], vec![0.0; n]], members: vec![], npairs: 0 }
    }
    fn base(&self, d: &Orbita, x: &[f64], p: [f64; 2]) -> Vec<f64> { match self.g0 { Some(g) => d.step(x, &context_of(g, p)), None => x.to_vec() } }
    /// añade pares (x_t → y_{t+1}) observados y reajusta W por ridge
    fn add(&mut self, d: &Orbita, x0: &[f64], ys: &[Vec<f64>], k: usize, p: [f64; 2], fam: usize) {
        let n = nfeat(self.deg);
        for t in 0..k {
            let x = if t == 0 { x0.to_vec() } else { ys[t - 1].clone() }; let y = &ys[t]; let bx = self.base(d, &x, p);
            for i in 0..x.len() / 2 {
                let f = phi(x[2 * i], x[2 * i + 1], p, self.deg);
                for r in 0..n { for c in 0..n { self.a[r * n + c] += f[r] * f[c]; } for o in 0..2 { self.b[o][r] += f[r] * (y[2 * i + o] - bx[2 * i + o]); } }
                self.npairs += 1;
            }
        }
        self.members.push(fam);
        let mut a = self.a.clone(); for r in 0..n { a[r * n + r] += self.lam; }
        for o in 0..2 { self.w[o] = solve(&a, &self.b[o], n); }
    }
    pub fn step(&self, d: &Orbita, x: &[f64], p: [f64; 2]) -> Vec<f64> {
        let mut y = self.base(d, x, p);
        for i in 0..x.len() / 2 {
            let f = phi(x[2 * i], x[2 * i + 1], p, self.deg);
            for o in 0..2 { y[2 * i + o] += f.iter().zip(&self.w[o]).map(|(a, b)| a * b).sum::<f64>(); }
        }
        y
    }
    /// sorpresa S = (1/k) Σ_{t≤k} min(rel_err(F^t(x0), y_t), 10)
    fn surprise(&self, d: &Orbita, x0: &[f64], ys: &[Vec<f64>], k: usize, p: [f64; 2]) -> f64 {
        let mut z = x0.to_vec(); let mut r = 0.0;
        for t in 0..k { z = self.step(d, &z, p); r += if fin(&z) { m::rel_err(&z, &ys[t]).min(10.0) } else { 10.0 }; }
        r / k as f64
    }
}

enum Pick { Known, Lib(usize) }
pub fn eval(a: &[String]) {
    let wpath = super::arg(a, "--weights").unwrap(); let wtxt = std::fs::read_to_string(&wpath).unwrap(); let d = Orbita::from_json(&wtxt);
    let k: usize = super::arg(a, "--k").unwrap().parse().unwrap(); let deg: usize = super::arg(a, "--deg").unwrap().parse().unwrap();
    let lam: f64 = super::arg(a, "--lam").unwrap().parse().unwrap(); let tau: f64 = super::arg(a, "--tau").unwrap().parse().unwrap();
    let set: Vec<(usize, usize, [f64; 2], Vec<f64>)> = std::fs::read_to_string(super::arg(a, "--set").unwrap()).unwrap().lines().skip(1).map(|l| {
        let f: Vec<&str> = l.split(',').collect(); (f[0].parse().unwrap(), f[1].parse().unwrap(), [f[2].parse().unwrap(), f[3].parse().unwrap()], nums(f[4], ';')) }).collect();
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let mut lib: Vec<Module> = vec![]; let mut ofam: std::collections::HashMap<usize, Module> = Default::default();
    struct Ep { idx: usize, tid: String, fam: usize, p: [f64; 2], x0: Vec<f64>, ys: Vec<Vec<f64>>, g: usize, pp: [f64; 2], sk: f64, row: String }
    let mut eps: Vec<Ep> = vec![];
    for line in std::fs::read_to_string(super::arg(a, "--ctx").unwrap()).unwrap().lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let idx: usize = f[0].parse().unwrap(); let c: Vec<f64> = f[3..13].iter().map(|v| v.parse().unwrap()).collect(); let ph = [c[8], c[9]];
        let (_, fam, ptrue, x0) = set.iter().find(|e| e.0 == idx).unwrap();
        let mut ys = vec![]; let mut y = x0.clone(); for _ in 0..16 { y = rule(*fam, *ptrue, &y); ys.push(y.clone()); }
        let ok = |z: &Vec<f64>, t: usize| fin(z) && m::correct(z, &ys[t - 1]);
        let hz = std::cell::RefCell::new(vec![]); // horizonte: nº de pasos consecutivos correctos tras t0 (exploratorio)
        let roll = |step: &dyn Fn(&[f64]) -> Vec<f64>, start: &Vec<f64>, t0: usize| { let mut z = start.clone(); let (mut o8, mut o16) = (false, false); let mut h = 0; let mut run = true;
            for t in t0 + 1..=16 { z = step(&z); if t == 8 { o8 = ok(&z, 8); } if t == 16 { o16 = ok(&z, 16); } if run && ok(&z, t) { h += 1; } else { run = false; } }
            hz.borrow_mut().push(h); (o8 as u8, o16 as u8) };
        let yk = &ys[k - 1];
        // BASE y OBSk
        let (b8, b16) = roll(&|z| d.step(z, &c), x0, 0);
        let (g, pp, r) = reid(&d, x0, &ys, k, ph); let sk = r / k as f64; let cg = context_of(g, pp);
        let (o8, o16) = roll(&|z| d.step(z, &cg), yk, k);
        // LEARNk
        let sl: Vec<f64> = lib.iter().map(|mm| mm.surprise(&d, x0, &ys, k, ph)).collect();
        let (mut best, mut pick) = (sk, Pick::Known);
        for (i, s) in sl.iter().enumerate() { if *s < best { best = *s; pick = Pick::Lib(i); } }
        let action; let mut mid: i64 = -1;
        if best <= tau {
            match pick { Pick::Known => { action = "known"; } Pick::Lib(i) => { lib[i].add(&d, x0, &ys, k, ph, *fam); action = "lib"; mid = i as i64; } }
        } else {
            let mut jb = (f64::INFINITY, usize::MAX, None);
            for (i, mm) in lib.iter().enumerate() { let mut m2 = mm.clone(); m2.add(&d, x0, &ys, k, ph, *fam); let s = m2.surprise(&d, x0, &ys, k, ph); if s < jb.0 { jb = (s, i, Some(m2)); } }
            if jb.0 <= tau { lib[jb.1] = jb.2.unwrap(); action = "join"; mid = jb.1 as i64; }
            else { let mut mm = Module::new(Some(g), deg, lam); mm.add(&d, x0, &ys, k, ph, *fam); lib.push(mm); action = "new"; mid = (lib.len() - 1) as i64; }
        }
        let hb = hz.borrow().clone(); let (hbase, hobs) = (hb[0], hb[1]);
        let (l8, l16) = if mid < 0 { (o8, o16) } else { let mm = &lib[mid as usize]; roll(&|z| mm.step(&d, z, ph), yk, k) };
        let hlearn = if mid < 0 { hobs } else { *hz.borrow().last().unwrap() };
        // LEARN_EPk: módulo sólo del episodio, descartado
        let (e8, e16, eact) = if sk <= tau { (o8, o16, "known") } else {
            let mut mm = Module::new(Some(g), deg, lam); mm.add(&d, x0, &ys, k, ph, *fam); let (a8, a16) = roll(&|z| mm.step(&d, z, ph), yk, k); (a8, a16, "new") };
        let hlep = if eact == "known" { hobs } else { *hz.borrow().last().unwrap() };
        // oráculos directos (sin núcleo): EP y FAM (agrupación por familia verdadera)
        let mut oe = Module::new(None, deg, lam); oe.add(&d, x0, &ys, k, ph, *fam); let (q8, q16) = roll(&|z| oe.step(&d, z, ph), yk, k);
        let of = ofam.entry(*fam).or_insert_with(|| Module::new(None, deg, lam)); of.add(&d, x0, &ys, k, ph, *fam);
        let of = of.clone(); let (f8, f16) = roll(&|z| of.step(&d, z, ph), yk, k);
        let mut pr: Vec<(&[f64], &[f64])> = vec![(x0, &ys[0])]; if k == 2 { pr.push((&ys[0], &ys[1])); }
        let wl = ls_fit(&pr); let (s8, s16) = roll(&|z| ls_step(&wl, z), yk, k);
        let row = format!("{idx},{},{fam},{sk:.8},{g},{b8},{b16},{o8},{o16},{l8},{l16},{action},{mid},{e8},{e16},{eact},{q8},{q16},{f8},{f16},{s8},{s16}", f[2]);
        let hv = hz.borrow().clone(); let n = hv.len(); let hzs = format!("{hbase},{hobs},{hlearn},{hlep},{},{},{}", hv[n - 3], hv[n - 2], hv[n - 1]);
        eps.push(Ep { idx, tid: f[2].to_string(), fam: *fam, p: ph, x0: x0.clone(), ys: ys.clone(), g, pp, sk, row: format!("{row}\t{hzs}") });
    }
    // FINALk: biblioteca congelada, argmin S entre conocido y módulos
    println!("pos,idx,text_id,fam,s_known,g_star,base_ok8,base_ok16,obs_ok8,obs_ok16,learn_ok8,learn_ok16,learn_action,learn_module,lep_ok8,lep_ok16,lep_action,oep_ok8,oep_ok16,ofam_ok8,ofam_ok16,ls_ok8,ls_ok16,final_ok8,final_ok16,final_pick,base_hz,obs_hz,learn_hz,lep_hz,oep_hz,ofam_hz,ls_hz");
    for (pos, e) in eps.iter().enumerate() {
        let _ = (&e.idx, &e.tid);
        let mut best = (e.sk, -1i64);
        for (i, mm) in lib.iter().enumerate() { let s = mm.surprise(&d, &e.x0, &e.ys, k, e.p); if s < best.0 { best = (s, i as i64); } }
        let ok = |z: &Vec<f64>, t: usize| fin(z) && m::correct(z, &e.ys[t - 1]);
        let mut z = e.ys[k - 1].clone(); let (mut o8, mut o16) = (false, false); let cg = context_of(e.g, e.pp);
        for t in k + 1..=16 { z = if best.1 < 0 { d.step(&z, &cg) } else { lib[best.1 as usize].step(&d, &z, e.p) }; if t == 8 { o8 = ok(&z, 8); } if t == 16 { o16 = ok(&z, 16); } }
        let (r0, h) = e.row.split_once('\t').unwrap();
        println!("{pos},{r0},{},{},{},{h}", o8 as u8, o16 as u8, best.1);
        let _ = e.fam;
    }
    let libj: Vec<_> = lib.iter().map(|mm| serde_json::json!({"g0": mm.g0, "members": mm.members, "npairs": mm.npairs})).collect();
    let after = Orbita::from_json(&wtxt); assert_eq!(after.raw, d.raw);
    eprintln!("queries={} n_modules={} library={}", cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(), lib.len(), serde_json::json!(libj));
}
