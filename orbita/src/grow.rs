//! ÓRBITA crece (docs/prereg_orbita_crece.md): expertos nuevos (observables no lineales aprendidos = subred por experto) añadidos a ÓRBITA
//! congelada cuando la sorpresa persiste; consolidación con replay, fusión/compactación y presupuesto de crecimiento.
//!   grow-eval --weights W --set S --ctx C(stream; cond = kN) --held H --hset HS --taus t2,t4,t8 --ce CE --rng-seed N
//!             --ckpts 0,150,... --held-out OUT --events EV                → stream CSV (stdout), held-out CSV, eventos JSONL
use crate::latent::{common, ctx8, fit_gn, lat_decide, lat_step, load_eps, load_set, npar, ok16, oep16, oms16, sur, Ctx8, Entry, Ep};
use crate::model::{CtxParams, Orbita};
use cdt_rqm_epr::v4_dataset::{context_of, N_FAM};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar as Rx;

pub const H_EXP: usize = 32; pub const E_MAX: usize = 8; pub const H_MONO: usize = 93;
const C_CONS: usize = 150; const N_GROW: usize = 15; const N_MIN: usize = 8; const N_ATT: usize = 3; const N_OOS: usize = 4; const CAP_BUF: usize = 300; const CAP_POOL: usize = 600; const CAP_MONO: usize = 1000;
const ST0: usize = 8000; const ST1: usize = 3000; const BATCH: usize = 128; const LR: f64 = 3e-3; const LR_END: f64 = 1e-4;
const LAM0: f64 = 1e-4; const LAMM: f64 = 1e-3; const ODEG: usize = 2; const OLAM: f64 = 1e-4; // fijados por 9166f1f

/// MLP din → H → H → 2 (tanh), Adam
#[derive(Clone)]
pub struct Mlp { din: usize, h: usize, w: Vec<f64>, m: Vec<f64>, v: Vec<f64>, t: i32 }
impl Mlp {
    pub fn npar(din: usize, h: usize) -> usize { h * din + h + h * h + h + 2 * h + 2 }
    fn new(din: usize, h: usize, rng: &mut Rx) -> Self {
        let n = Self::npar(din, h); let mut w = vec![0.0; n]; let mut o = 0;
        for (fi, fo, sc) in [(din, h, 1.0), (h, h, 1.0), (h, 2, 0.1)] {
            let a = sc * (6.0 / (fi + fo) as f64).sqrt(); for i in 0..fi * fo { w[o + i] = rng.gen_range(-a..a); } o += fi * fo + fo; }
        Self { din, h, w, m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }
    fn offs(&self) -> [usize; 6] { let (d, h) = (self.din, self.h); let a = h * d; let b = a + h; let c = b + h * h; let e = c + h; let f = e + 2 * h; [0, a, b, c, e, f] }
    fn fwd(&self, x: &[f64]) -> ([f64; 2], Vec<f64>, Vec<f64>) {
        let (d, h) = (self.din, self.h); let o = self.offs(); let w = &self.w;
        let a1: Vec<f64> = (0..h).map(|i| (w[o[1] + i] + (0..d).map(|j| w[o[0] + i * d + j] * x[j]).sum::<f64>()).tanh()).collect();
        let a2: Vec<f64> = (0..h).map(|i| (w[o[3] + i] + (0..h).map(|j| w[o[2] + i * h + j] * a1[j]).sum::<f64>()).tanh()).collect();
        let y = [0, 1].map(|i| w[o[5] + i] + (0..h).map(|j| w[o[4] + i * h + j] * a2[j]).sum::<f64>()); (y, a1, a2)
    }
    pub fn out(&self, x: &[f64]) -> [f64; 2] { self.fwd(x).0 }
    /// pérdida Σ w_i ‖f(x_i) − y_i‖², minibatches aleatorios
    fn train(&mut self, xs: &[Vec<f64>], ys: &[[f64; 2]], ws: &[f64], steps: usize, rng: &mut Rx) {
        if xs.is_empty() { return; }
        let (d, h) = (self.din, self.h); let o = self.offs(); let n = self.w.len();
        self.m.iter_mut().for_each(|v| *v = 0.0); self.v.iter_mut().for_each(|v| *v = 0.0); self.t = 0;
        for it in 0..steps {
            let lr = LR_END + 0.5 * (LR - LR_END) * (1.0 + (std::f64::consts::PI * it as f64 / steps as f64).cos());
            let mut g = vec![0.0; n]; let bs = BATCH.min(xs.len());
            for _ in 0..bs {
                let s = rng.gen_range(0..xs.len()); let (y, a1, a2) = self.fwd(&xs[s]); let w = &self.w;
                let d3 = [0, 1].map(|i| 2.0 * ws[s] * (y[i] - ys[s][i]));
                let mut dz2 = vec![0.0; h];
                for i in 0..2 { g[o[5] + i] += d3[i]; for j in 0..h { g[o[4] + i * h + j] += d3[i] * a2[j]; dz2[j] += w[o[4] + i * h + j] * d3[i]; } }
                for j in 0..h { dz2[j] *= 1.0 - a2[j] * a2[j]; }
                let mut dz1 = vec![0.0; h];
                for i in 0..h { g[o[3] + i] += dz2[i]; for j in 0..h { g[o[2] + i * h + j] += dz2[i] * a1[j]; dz1[j] += w[o[2] + i * h + j] * dz2[i]; } }
                for j in 0..h { dz1[j] *= 1.0 - a1[j] * a1[j]; }
                for i in 0..h { g[o[1] + i] += dz1[i]; for j in 0..d { g[o[0] + i * d + j] += dz1[i] * xs[s][j]; } }
            }
            self.t += 1; let (b1, b2) = (0.9f64, 0.999f64);
            for i in 0..n { let gi = g[i] / bs as f64; self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi; self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
                let mh = self.m[i] / (1.0 - b1.powi(self.t)); let vh = self.v[i] / (1.0 - b2.powi(self.t)); self.w[i] -= lr * mh / (vh.sqrt() + 1e-8); }
        }
    }
}
fn orb(d: &Orbita, cp: &CtxParams, c: &[f64], x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for i in 0..x.len() / 2 { let z = d.encode(&x[2 * i..2 * i + 2], c); let w = d.correct(cp, &d.core(cp, &z)); out[2 * i] = w[0]; out[2 * i + 1] = w[1]; }
    out
}
/// episodio observado guardado (sólo y_1..y_k)
struct St { x0: Vec<f64>, ys: Vec<Vec<f64>>, k: usize, ph: [f64; 2], gs: usize, fam: usize }
/// tipo de red: Expert (base ÓRBITA c(g0, p̂)), Mono (base ÓRBITA c(g*, p̂), entrada + onehot g*), Fam (base identidad; oráculo por familia)
#[derive(Clone, Copy, PartialEq)] enum Kind { Expert(usize), Mono, Fam }
#[derive(Clone)]
struct Net { kind: Kind, net: Mlp, buf: Vec<usize>, seen: usize, trained: bool, id: usize, used: usize, born: usize }
fn inp(kind: Kind, q: &[f64], ph: [f64; 2], gs: usize) -> Vec<f64> {
    let mut v = vec![q[0] / 2.0, q[1] / 2.0, ph[0], ph[1]]; if kind == Kind::Mono { for g in 0..N_FAM { v.push((g == gs) as u8 as f64); } } v
}
impl Net {
    fn new(kind: Kind, rng: &mut Rx, id: usize) -> Self { let (din, h) = if kind == Kind::Mono { (4 + N_FAM, H_MONO) } else { (4, H_EXP) }; Net { kind, net: Mlp::new(din, h, rng), buf: vec![], seen: 0, trained: false, id, used: 0, born: 0 } }
    fn g0(&self, gs: usize) -> Option<usize> { match self.kind { Kind::Expert(g) => Some(g), Kind::Mono => Some(gs), Kind::Fam => None } }
    fn step(&self, d: &Orbita, cx: &Ctx8, x: &[f64], ph: [f64; 2], gs: usize) -> Vec<f64> {
        let mut y = match self.g0(gs) { Some(g) => orb(d, &cx.cps[g], &cx.cs[g], x), None => x.to_vec() };
        for i in 0..x.len() / 2 { let r = self.net.out(&inp(self.kind, &x[2 * i..2 * i + 2], ph, gs)); y[2 * i] += r[0]; y[2 * i + 1] += r[1]; }
        y
    }
    fn s(&self, d: &Orbita, cx: &Ctx8, x0: &[f64], ys: &[Vec<f64>], k: usize, ph: [f64; 2], gs: usize) -> f64 { sur(&|z| self.step(d, cx, z, ph, gs), x0, ys, k) }
    /// reservoir: conserva una muestra uniforme de tamaño ≤ cap
    fn push(&mut self, id: usize, cap: usize, rng: &mut Rx) { self.seen += 1; if self.buf.len() < cap { self.buf.push(id); } else { let j = rng.gen_range(0..self.seen); if j < cap { self.buf[j] = id; } } }
    fn fit(&mut self, d: &Orbita, st: &[St], ids: &[usize], steps: usize, rng: &mut Rx) {
        let (mut xs, mut ys, mut ws) = (vec![], vec![], vec![]);
        for &i in ids { let e = &st[i]; let g = self.g0(e.gs); let cp = g.map(|g| d.ctx(&context_of(g, e.ph))); let c = g.map(|g| context_of(g, e.ph));
            for t in 0..e.k { let x = if t == 0 { &e.x0 } else { &e.ys[t - 1] }; let y = &e.ys[t];
                let b = match g { Some(_) => orb(d, cp.as_ref().unwrap(), c.as_ref().unwrap(), x), None => x.clone() }; let wn = 1.0 / y.iter().map(|v| v * v).sum::<f64>().max(1e-9);
                for p in 0..x.len() / 2 { xs.push(inp(self.kind, &x[2 * p..2 * p + 2], e.ph, e.gs)); ys.push([y[2 * p] - b[2 * p], y[2 * p + 1] - b[2 * p + 1]]); ws.push(wn); } } }
        self.net.train(&xs, &ys, &ws, steps, rng); self.trained = true;
    }
}
fn s_st(d: &Orbita, n: &Net, e: &St) -> f64 { let cx = ctx8(d, e.ph); n.s(d, &cx, &e.x0, &e.ys, e.k, e.ph, e.gs) }
fn mode(v: impl Iterator<Item = usize>) -> usize { let mut c = [0usize; 16]; for x in v { c[x] += 1; } (0..16).max_by_key(|&i| c[i]).unwrap() }
fn argf(a: &[String], k: &str) -> f64 { super::arg(a, k).unwrap().parse().unwrap() }
fn nums(s: &str) -> Vec<f64> { s.split(',').map(|t| t.trim().parse().unwrap()).collect() }

pub fn run(a: &[String]) {
    let wtxt = std::fs::read_to_string(super::arg(a, "--weights").unwrap()).unwrap(); let d = Orbita::from_json(&wtxt);
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let taus = nums(&super::arg(a, "--taus").unwrap()); let tau_of = |k: usize| match k { 2 => taus[0], 4 => taus[1], _ => taus[2] }; let ce = argf(a, "--ce");
    let mut rng = Rx::seed_from_u64(argf(a, "--rng-seed") as u64 ^ 0x6A0F);
    let set = load_set(&super::arg(a, "--set").unwrap()); let stream = load_eps(&super::arg(a, "--ctx").unwrap(), &set);
    let hset = load_set(&super::arg(a, "--hset").unwrap()); let held = load_eps(&super::arg(a, "--held").unwrap(), &hset);
    let ckpts: Vec<usize> = nums(&super::arg(a, "--ckpts").unwrap()).iter().map(|v| *v as usize).collect();
    let nfam = 1 + set.iter().map(|s| s.1).max().unwrap();
    let mut st: Vec<St> = vec![]; let mut experts: Vec<Net> = vec![]; let mut pool: Vec<usize> = vec![]; let mut nid = 0usize;
    let mut mono = Net::new(Kind::Mono, &mut rng, 999); let mut fams: Vec<Net> = (0..nfam).map(|f| Net::new(Kind::Fam, &mut rng, 1000 + f)).collect();
    let mut lib: Vec<Entry> = vec![]; let mut ev = String::new(); let mut nnew = 0usize; // nuevos datos desde la última consolidación (por red)
    let mut dirty: std::collections::HashSet<usize> = Default::default();
    // precálculo held-out por (episodio, k)
    let hk: Vec<Vec<(f64, usize, u8, u8, u8, u8)>> = held.iter().map(|(e, _)| [2usize, 4, 8].iter().map(|&k| {
        let (sk, gs, _, obs) = common(&d, e, k); let cx = ctx8(&d, e.ph);
        let lep = if sk <= tau_of(k) { obs } else { let aa = fit_gn(&|th, z| lat_step(&d, &cx.cps[gs], &cx.cs[gs], th, z), &vec![0.0; npar()], LAM0, &e.x0, &e.ys, k); ok16(e, &|z| lat_step(&d, &cx.cps[gs], &cx.cs[gs], &aa, z), k) };
        (sk, gs, obs, lep, oep16(&d, e, k, ODEG, OLAM), oms16(e, k, ODEG, OLAM)) }).collect()).collect();
    // routing común: (predicción ok16, acción, id experto)
    let route = |nets: &[&Net], e: &Ep, cx: &Ctx8, k: usize, sk: f64, gs: usize, obs: u8, lep: u8| -> (u8, &'static str, i64) {
        let mut b = (f64::INFINITY, usize::MAX);
        for (i, n) in nets.iter().enumerate() { if !n.trained { continue; } let s = n.s(&d, cx, &e.x0, &e.ys, k, e.ph, gs); if s < b.0 { b = (s, i); } }
        if sk <= tau_of(k) && sk <= b.0 { return (obs, "known", -1); }
        if b.0 <= ce * tau_of(k) { let n = nets[b.1]; return (ok16(e, &|z| n.step(&d, cx, z, e.ph, gs), k), "net", b.1 as i64); }
        (lep, "fallback", -1)
    };
    let mut hout = String::from("ckpt,idx,text_id,fam,k,obs16,latep16,oep16,oms16,lat16,grow16,grow_action,grow_expert_fam,mono16,mono_action,famnet16,n_experts\n");
    let mut sout = String::from("pos,idx,text_id,fam,k,s_known,obs16,latep16,lat16,grow16,grow_action,grow_expert_fam,mono16,famnet16,n_experts,pool,lib_size\n");
    let maj = |n: &Net, st: &[St]| -> i64 { if n.buf.is_empty() { -1 } else { mode(n.buf.iter().map(|&i| st[i].fam)) as i64 } };
    let total = stream.len();
    for pos in 0..=total {
        if ckpts.contains(&pos) {
            let er: Vec<&Net> = experts.iter().collect(); let final_ck = pos == *ckpts.iter().max().unwrap();
            for (j, (e, _)) in held.iter().enumerate() { let cx = ctx8(&d, e.ph);
                for (ki, &k) in [2usize, 4, 8].iter().enumerate() { let (sk, gs, obs, lep, oe, om) = hk[j][ki];
                    let (g16, ga, gi) = route(&er, e, &cx, k, sk, gs, obs, lep); let gf = if gi >= 0 { maj(er[gi as usize], &st) } else { -1 };
                    let (m16, ma, _) = route(&[&mono], e, &cx, k, sk, gs, obs, lep);
                    let fnn = &fams[e.fam]; let f16 = if fnn.trained { ok16(e, &|z| fnn.step(&d, &cx, z, e.ph, gs), k) } else { lep };
                    let l16 = if pos == 0 { lep.to_string() } else if final_ck { let (dec, _, _) = lat_decide(&d, &cx, &lib, e, k, sk, gs, tau_of(k), LAM0, LAMM);
                        (match &dec { None => obs, Some((g0, aa)) => ok16(e, &|z| lat_step(&d, &cx.cps[*g0], &cx.cs[*g0], aa, z), k) }).to_string() } else { String::new() };
                    hout += &format!("{pos},{},{},{},{k},{obs},{lep},{oe},{om},{l16},{g16},{ga},{gf},{m16},{ma},{f16},{}\n", e.idx, e.tid, e.fam, experts.len()); } }
        }
        if pos == total { break; }
        let (e, cond) = &stream[pos]; let k: usize = cond.trim_start_matches('k').parse().unwrap(); let tau = tau_of(k);
        let (sk, gs, _, obs) = common(&d, e, k); let cx = ctx8(&d, e.ph);
        let lep = if sk <= tau { obs } else { let aa = fit_gn(&|th, z| lat_step(&d, &cx.cps[gs], &cx.cs[gs], th, z), &vec![0.0; npar()], LAM0, &e.x0, &e.ys, k); ok16(e, &|z| lat_step(&d, &cx.cps[gs], &cx.cs[gs], &aa, z), k) };
        let (dec, _, _) = lat_decide(&d, &cx, &lib, e, k, sk, gs, tau, LAM0, LAMM);
        let l16 = match &dec { None => obs, Some((g0, aa)) => ok16(e, &|z| lat_step(&d, &cx.cps[*g0], &cx.cs[*g0], aa, z), k) };
        if let Some((g0, aa)) = dec { lib.push(Entry { g0, a: aa, fam: e.fam }); }
        let er: Vec<&Net> = experts.iter().collect(); let (g16, ga, gi) = route(&er, e, &cx, k, sk, gs, obs, lep); let gf = if gi >= 0 { maj(er[gi as usize], &st) } else { -1 };
        let (m16, _, _) = route(&[&mono], e, &cx, k, sk, gs, obs, lep);
        let f16 = { let fnn = &fams[e.fam]; if fnn.trained { ok16(e, &|z| fnn.step(&d, &cx, z, e.ph, gs), k) } else { lep } };
        sout += &format!("{pos},{},{},{},{k},{sk:.8},{obs},{lep},{l16},{g16},{ga},{gf},{m16},{f16},{},{},{}\n", e.idx, e.tid, e.fam, experts.len(), pool.len(), lib.len());
        // guardar lo observado (sólo y_1..y_k)
        let id = st.len(); st.push(St { x0: e.x0.clone(), ys: e.ys[..k].to_vec(), k, ph: e.ph, gs, fam: e.fam }); nnew += 1;
        fams[e.fam].push(id, CAP_BUF, &mut rng); dirty.insert(1000 + e.fam);
        if sk > tau { mono.push(id, CAP_MONO, &mut rng); dirty.insert(999); }
        if ga == "net" { experts[gi as usize].used += 1; experts[gi as usize].push(id, CAP_BUF, &mut rng); dirty.insert(experts[gi as usize].id); }
        else if sk > tau { pool.push(id); if pool.len() > CAP_POOL { pool.remove(0); } }
        // consolidación
        if (pos + 1) % C_CONS == 0 {
            let _ = nnew; nnew = 0;
            for x in experts.iter_mut() { if dirty.contains(&x.id) { let ids = x.buf.clone(); x.fit(&d, &st, &ids, ST1, &mut rng); } }
            if dirty.contains(&999) { let ids = mono.buf.clone(); let s = if mono.trained { ST1 } else { ST0 }; mono.fit(&d, &st, &ids, s, &mut rng); }
            for f in fams.iter_mut() { if dirty.contains(&f.id) { let ids = f.buf.clone(); let s = if f.trained { ST1 } else { ST0 }; f.fit(&d, &st, &ids, s, &mut rng); } }
            dirty.clear();
            // crecimiento (sorpresa persistente: pool ≥ N_GROW)
            // semilla = mayor grupo del pool por familia identificada g* (no se usa la familia real); se prueba cada g* como mucho una vez
            // presupuesto: si está lleno, desalojar el experto sin uso en el stream desde su creación (edad ≥ 2 consolidaciones), el de menor uso
            if experts.len() >= E_MAX && pool.len() >= N_GROW {
                if let Some((j, _)) = experts.iter().enumerate().filter(|(_, x)| x.used == 0 && pos + 1 >= x.born + 2 * C_CONS).min_by_key(|(_, x)| x.born) {
                    ev += &format!("{{\"pos\":{pos},\"event\":\"evict\",\"id\":{},\"n\":{}}}\n", experts[j].id, experts[j].buf.len()); experts.remove(j); } }
            let mut tried: Vec<usize> = vec![]; let mut att = 0;
            while pool.len() >= N_GROW && experts.len() < E_MAX && att < N_ATT {
                let mut cnt = [0usize; 16]; for &i in &pool { cnt[st[i].gs] += 1; }
                let g0 = match (0..16).filter(|g| !tried.contains(g) && cnt[*g] >= N_MIN).max_by_key(|&g| (cnt[g], 16 - g)) { Some(g) => g, None => break };
                tried.push(g0); att += 1; let mut x = Net::new(Kind::Expert(g0), &mut rng, nid); nid += 1;
                // ajuste sólo con 2/3 del grupo semilla; el resto del pool queda fuera de muestra
                let mut p0: Vec<usize> = pool.iter().copied().filter(|&i| st[i].gs == g0).collect(); for t in 0..p0.len() { let s = rng.gen_range(t..p0.len()); p0.swap(t, s); } p0.truncate((2 * p0.len() / 3).max(N_MIN));
                if std::env::var("GROW_DEBUG").is_ok() { eprintln!("pool fam,gs {:?}", pool.iter().map(|&i| (st[i].fam, st[i].gs)).collect::<Vec<_>>()); }
                let dbg = |x: &Net, tag: &str| if std::env::var("GROW_DEBUG").is_ok() { let mut v: Vec<f64> = p0.iter().map(|&i| s_st(&d, x, &st[i]) / tau_of(st[i].k)).collect(); v.sort_by(|a, b| a.total_cmp(b)); eprintln!("{tag} S/tau q10={:.2} q50={:.2} q90={:.2}", v[v.len() / 10], v[v.len() / 2], v[9 * v.len() / 10]); };
                dbg(&x, "init"); x.fit(&d, &st, &p0, std::env::var("GROW_ST0").map(|v| v.parse().unwrap()).unwrap_or(ST0), &mut rng); dbg(&x, "fit");
                for _ in 0..2 { let mut sc: Vec<(f64, usize)> = p0.iter().map(|&i| (s_st(&d, &x, &st[i]), i)).collect(); sc.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let half: Vec<usize> = sc.iter().take((sc.len() / 2).max(N_MIN)).map(|v| v.1).collect(); x.fit(&d, &st, &half, ST1, &mut rng); dbg(&x, "trim"); }
                let claim: Vec<usize> = pool.iter().cloned().filter(|&i| s_st(&d, &x, &st[i]) <= ce * tau_of(st[i].k)).collect();
                let fc: Vec<usize> = claim.iter().map(|&i| st[i].fam).collect();
                let oos = claim.iter().filter(|i| !p0.contains(i)).count();
                if claim.len() >= N_MIN && oos >= N_OOS { x.born = pos + 1; x.fit(&d, &st, &claim, ST1, &mut rng); for &i in &claim { x.push(i, CAP_BUF, &mut rng); } pool.retain(|i| !claim.contains(i));
                    ev += &format!("{{\"pos\":{pos},\"event\":\"grow\",\"id\":{},\"g0\":{g0},\"claimed\":{},\"oos\":{oos},\"fams\":{:?}}}\n", x.id, claim.len(), fc); experts.push(x); }
                else { ev += &format!("{{\"pos\":{pos},\"event\":\"grow_fail\",\"g0\":{g0},\"claimed\":{},\"oos\":{oos},\"pool\":{}}}\n", claim.len(), pool.len()); }
            }
            // fusión/compactación (como mucho una por consolidación)
            'm: for i in 0..experts.len() { for j in i + 1..experts.len() {
                let rj: Vec<usize> = experts[j].buf.iter().rev().take(10).cloned().collect(); let ri: Vec<usize> = experts[i].buf.iter().rev().take(10).cloned().collect();
                let ok_ij = rj.iter().filter(|&&q| s_st(&d, &experts[i], &st[q]) <= 2.0 * ce * tau_of(st[q].k)).count() * 2 >= rj.len();
                let ok_ji = ri.iter().filter(|&&q| s_st(&d, &experts[j], &st[q]) <= 2.0 * ce * tau_of(st[q].k)).count() * 2 >= ri.len();
                if !(ok_ij || ok_ji) { continue; }
                let rate = |n: &Net, ids: &[usize]| ids.iter().filter(|&&q| s_st(&d, n, &st[q]) <= ce * tau_of(st[q].k)).count() as f64 / ids.len().max(1) as f64;
                let r0 = 0.5 * (rate(&experts[i], &experts[i].buf) + rate(&experts[j], &experts[j].buf));
                let mut u: Vec<usize> = experts[i].buf.clone(); u.extend(experts[j].buf.iter()); if u.len() > CAP_BUF { for t in 0..u.len() { let s = rng.gen_range(t..u.len()); u.swap(t, s); } u.truncate(CAP_BUF); }
                let mut mg = if experts[i].buf.len() >= experts[j].buf.len() { experts[i].clone() } else { experts[j].clone() }; mg.fit(&d, &st, &u, 1000, &mut rng);
                let r1 = rate(&mg, &u);
                if r1 >= 0.9 * r0 { ev += &format!("{{\"pos\":{pos},\"event\":\"merge\",\"a\":{},\"b\":{},\"rate_before\":{r0:.3},\"rate_after\":{r1:.3}}}\n", experts[i].id, experts[j].id);
                    mg.buf = u; mg.seen = experts[i].seen + experts[j].seen; mg.used = experts[i].used + experts[j].used; mg.born = experts[i].born.min(experts[j].born); mg.id = nid; nid += 1; experts.remove(j); experts[i] = mg; break 'm; }
            } }
            let sum: Vec<String> = experts.iter().map(|x| { let mut c = vec![0usize; nfam]; for &i in &x.buf { c[st[i].fam] += 1; } format!("{{\"id\":{},\"g0\":{},\"n\":{},\"seen\":{},\"used\":{},\"fams\":{:?}}}", x.id, x.g0(0).unwrap(), x.buf.len(), x.seen, x.used, c) }).collect();
            ev += &format!("{{\"pos\":{pos},\"event\":\"consolidate\",\"pool\":{},\"params\":{},\"experts\":[{}]}}\n", pool.len(), experts.len() * Mlp::npar(4, H_EXP), sum.join(","));
        }
    }
    print!("{sout}"); std::fs::write(super::arg(a, "--held-out").unwrap(), hout).unwrap(); std::fs::write(super::arg(a, "--events").unwrap(), ev).unwrap();
    assert_eq!(Orbita::from_json(&wtxt).raw, d.raw);
    eprintln!("queries={} n_experts={} params_grow_max={} params_mono={}", cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(), experts.len(), E_MAX * Mlp::npar(4, H_EXP), Mlp::npar(4 + N_FAM, H_MONO));
}
