//! E0–E2 memoria por capas (docs/preregistro_memoria_capas.md). Usa v4 sin modificarlo. Nunca lee ds.test.
use crate::model::Orbita;
use cdt_rqm_epr::field_autonomy_v4::*;
use cdt_rqm_epr::v4_dataset::{context_of, generate_and_seal, l2, DatasetConfig, Example, ObjectClass, Partition, N_FAM};
use cdt_rqm_epr::v4_distill::{family_fit_residual, TRUST_TAU};
use cdt_rqm_epr::v4_metrics as m;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

const H: usize = 4;
const W: usize = 64;
const RHO: f64 = 0.25;
const NMAX: usize = 256;
const BASE: [usize; 2] = [1, 6];

fn rot(t: f64, x: f64, y: f64) -> (f64, f64) { (t.cos() * x - t.sin() * y, t.sin() * x + t.cos() * y) }
pub fn b_point(b: usize, p: [f64; 2], x: f64, y: f64) -> (f64, f64) {
    if b == 0 {
        let (u, v) = rot(0.6 * p[0], x, y);
        (u + 0.25 * p[1] * (2.0 * v).sin(), v + 0.25 * p[1] * (2.0 * u).sin())
    } else {
        let r = (x * x + y * y).sqrt();
        let (u, v) = rot(0.4 * p[0] + 0.8 * p[1] * (r - 0.7), x, y);
        let s = (0.1 * p[1]).exp();
        (s * u, s * v)
    }
}
fn b_apply(b: usize, p: [f64; 2], o: &[f64]) -> Vec<f64> {
    let mut v = o.to_vec();
    for i in 0..o.len() / 2 { let (a, c) = b_point(b, p, o[2 * i], o[2 * i + 1]); v[2 * i] = a; v[2 * i + 1] = c; }
    v
}

#[derive(Clone)]
pub struct Ep { pub set: char, pub truefam: usize, pub fam: usize, pub p: [f64; 2], pub x0: Vec<f64>, pub ys: Vec<Vec<f64>> }

fn ep_a(e: &Example) -> Ep {
    let mut ys = vec![]; let mut y = e.input.clone();
    for _ in 0..H { y = cdt_rqm_epr::v4_dataset::apply_rule(e.family_id, e.params, &y); ys.push(y.clone()); }
    Ep { set: 'A', truefam: e.family_id, fam: e.family_id, p: e.params, x0: e.input.clone(), ys }
}
fn ep_b(b: usize, p: [f64; 2], x0: &[f64]) -> Ep {
    let mut ys = vec![]; let mut y = x0.to_vec();
    for _ in 0..H { y = b_apply(b, p, &y); ys.push(y.clone()); }
    Ep { set: 'B', truefam: 8 + b, fam: BASE[b], p, x0: x0.to_vec(), ys }
}
fn as_example(e: &Ep) -> Example {
    Example { rule_id: e.fam, family_id: e.fam, instance_id: 0, object_class: ObjectClass::Blob, params: e.p,
        canonical_target_hash: 0, input: e.x0.clone(), context: context_of(e.fam, e.p), expected: e.ys[0].clone(),
        partition: Partition::Train, subset: "mem" }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Var { Cdt, CdtNp, CdtGen, Shuf, Corrupt, CdtRand, CdtFixed }

fn consolidate_v(v: Var, ex: &[Example], seed: u64) -> LearningSignal {
    let mode = if v == Var::Shuf { ExpMode::ShuffledOrder } else { ExpMode::Real };
    let store = collect_experience(ex, mode, seed);
    let mut sig = if v == Var::CdtGen {
        consolidate_generic(store.read_all(), ConsolidateMode::Full, seed)
    } else {
        consolidate(&store, if v == Var::Corrupt { ConsolidateMode::Corrupt } else { ConsolidateMode::Full })
    };
    if matches!(v, Var::Cdt | Var::Shuf | Var::Corrupt | Var::CdtFixed) {
        let res = family_fit_residual(&sig, store.read_all());
        sig.rules.retain(|r| res[r.fam] < TRUST_TAU);
    }
    if v == Var::CdtRand { // E5: poda aleatoria del mismo número de reglas que la poda trusted
        let res = family_fit_residual(&sig, store.read_all());
        let k = sig.rules.iter().filter(|r| res[r.fam] >= TRUST_TAU).count();
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x9A4D);
        for _ in 0..k { let j = rng.gen_range(0..sig.rules.len()); sig.rules.remove(j); }
    }
    sig
}
fn has(sig: &Option<LearningSignal>, f: usize) -> bool {
    match sig { None => false, Some(s) => match &s.generic { Some(g) => g.fams.iter().any(|x| x.fam == f), None => s.rules.iter().any(|r| r.fam == f) } }
}
fn nnum(sig: &Option<LearningSignal>) -> usize { sig.as_ref().map(|s| if s.rules.is_empty() && s.generic.is_none() { 0 } else { s.n_numbers() }).unwrap_or(0) }

#[derive(Clone, Copy, PartialEq)]
enum Route { A, Orb, B }

struct Mem { v: Var, a: Option<LearningSignal>, b: Option<LearningSignal>, tau: [f64; N_FAM], nprev: usize, sleeps: usize }
impl Mem {
    fn one(&self, d: &Orbita, rt: Route, f: usize, p: [f64; 2], x: &[f64]) -> Vec<f64> {
        match rt {
            Route::A => self.a.as_ref().unwrap().predict(f, p, x).unwrap(),
            Route::B => self.b.as_ref().unwrap().predict(f, p, x).unwrap(),
            Route::Orb => d.step(x, &context_of(f, p)),
        }
    }
    fn active(&self, f: usize) -> Route { if has(&self.a, f) { Route::A } else { Route::Orb } }
    /// (ruta de evaluación, r, novel)
    fn route(&self, d: &Orbita, e: &Ep) -> (Route, f64, bool) {
        let ra = self.active(e.fam);
        let r = m::rel_err(&self.one(d, ra, e.fam, e.p, &e.x0), &e.ys[0]);
        if r <= self.tau[e.fam] { return (ra, r, false); }
        if has(&self.b, e.fam) {
            let rb = m::rel_err(&self.one(d, Route::B, e.fam, e.p, &e.x0), &e.ys[0]);
            return (Route::B, r, rb > self.tau[e.fam]);
        }
        (Route::Orb, r, true)
    }
    fn sleep_b(&mut self, buf: &mut Vec<Ep>, seed: u64) {
        if buf.is_empty() { return; }
        let mut ex: Vec<Example> = buf.iter().map(as_example).collect();
        if let Some(prev) = &self.b { ex.extend(pseudo_experiences(prev, self.nprev, 6, seed ^ self.sleeps as u64)); }
        self.nprev = ex.len();
        let s = consolidate_v(self.v, &ex, seed ^ 0xB5EE ^ self.sleeps as u64);
        self.b = if s.rules.is_empty() && s.generic.is_none() { None } else { Some(s) };
        self.sleeps += 1;
        buf.clear();
    }
}

fn rollout<F: Fn(&[f64]) -> Vec<f64>>(f: F, e: &Ep) -> (bool, bool) {
    let mut y = e.x0.clone(); let mut ok = vec![];
    for h in 0..H { y = f(&y); ok.push(m::correct(&y, &e.ys[h])); }
    (ok[0], ok[H - 1])
}
fn ls_affine(e: &Ep) -> [[f64; 3]; 2] {
    let k = e.x0.len() / 2; let mut a = [[0.0; 3]; 3]; let mut b = [[0.0; 3]; 2];
    for i in 0..k {
        let u = [e.x0[2 * i], e.x0[2 * i + 1], 1.0];
        for r in 0..3 { for c in 0..3 { a[r][c] += u[r] * u[c]; } for o in 0..2 { b[o][r] += u[r] * e.ys[0][2 * i + o]; } }
    }
    for r in 0..3 { a[r][r] += 1e-9; }
    let mut out = [[0.0; 3]; 2];
    for o in 0..2 { // Gauss
        let mut m_ = a; let mut v = b[o];
        for c in 0..3 { let piv = (c..3).max_by(|&i, &j| m_[i][c].abs().total_cmp(&m_[j][c].abs())).unwrap(); m_.swap(c, piv); v.swap(c, piv);
            for r in 0..3 { if r != c { let f = m_[r][c] / m_[c][c]; for cc in 0..3 { m_[r][cc] -= f * m_[c][cc]; } v[r] -= f * v[c]; } } }
        for c in 0..3 { out[o][c] = v[c] / m_[c][c]; }
    }
    out
}

struct Raw { cap: usize, seen: usize, items: Vec<(usize, bool, Ep)> } // (fam, slotB, ep)
impl Raw {
    fn push(&mut self, rng: &mut Xoshiro256StarStar, it: (usize, bool, Ep)) {
        self.seen += 1;
        if self.items.len() < self.cap { self.items.push(it); } else if self.cap > 0 { let j = rng.gen_range(0..self.seen); if j < self.cap { self.items[j] = it; } }
    }
    fn step(&self, f: usize, slot_b: bool, p: [f64; 2], x: &[f64]) -> Option<Vec<f64>> {
        let c = context_of(f, p);
        let best = self.items.iter().filter(|(ff, sb, _)| *ff == f && *sb == slot_b)
            .map(|(_, _, e)| (l2(&e.x0, x).powi(2) + 4.0 * l2(&context_of(e.fam, e.p), &c).powi(2), e))
            .min_by(|a, b| a.0.total_cmp(&b.0))?.1;
        Some(x.iter().zip(&best.x0).zip(&best.ys[0]).map(|((a, b), y)| a + (y - b)).collect())
    }
}

pub fn run(a: &[String], seed: u64, d: &Orbita) {
    let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS));
    let a_wake: Vec<Ep> = s.ds.train.iter().map(ep_a).collect();
    let a_eval: Vec<Ep> = s.ds.dev.iter().map(ep_a).collect();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xB0B);
    let mut b_wake = vec![];
    for i in 0..1024 { let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]; b_wake.push(ep_b(i % 2, p, &s.ds.train[(i * 7919) % s.ds.train.len()].input)); }
    let mut b_eval = vec![];
    for i in 0..512 { let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]; b_eval.push(ep_b(i % 2, p, &s.ds.dev[(i * 104729) % s.ds.dev.len()].input)); }
    let evals: Vec<&Ep> = a_eval.iter().chain(&b_eval).collect();
    let stage2 = a.iter().any(|x| x == "--stage2");
    println!("set,truefam,fam,cond,ok1,ok4,r,slot,p1,p2");
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    // controles sin memoria
    for e in evals.iter().filter(|_| !stage2) {
        let (o1, o4) = rollout(|x| d.step(x, &context_of(e.fam, e.p)), e);
        println!("{},{},{},NOCDT,{},{},", e.set, e.truefam, e.fam, o1 as u8, o4 as u8);
        let fb = (0..N_FAM).min_by(|&i, &j| m::rel_err(&d.step(&e.x0, &context_of(i, e.p)), &e.ys[0]).total_cmp(&m::rel_err(&d.step(&e.x0, &context_of(j, e.p)), &e.ys[0]))).unwrap();
        let (o1, o4) = rollout(|x| d.step(x, &context_of(fb, e.p)), e);
        println!("{},{},{},ORACLE_ORB,{},{},", e.set, e.truefam, e.fam, o1 as u8, o4 as u8);
        let w = ls_affine(e);
        let (o1, o4) = rollout(|x| { let mut y = x.to_vec(); for i in 0..x.len() / 2 { for o in 0..2 { y[2 * i + o] = w[o][0] * x[2 * i] + w[o][1] * x[2 * i + 1] + w[o][2]; } } y }, e);
        println!("{},{},{},ORACLE_LS,{},{},", e.set, e.truefam, e.fam, o1 as u8, o4 as u8);
    }
    let mut meta = vec![];
    let vars: Vec<Var> = if stage2 { vec![Var::Cdt, Var::CdtFixed, Var::CdtRand, Var::Shuf] } else { vec![Var::Cdt, Var::CdtNp, Var::CdtGen, Var::Shuf, Var::Corrupt] };
    let mut n_b_sleeps = 1usize;
    for v in vars {
        let name = match v { Var::Cdt => "CDT", Var::CdtNp => "CDT_NP", Var::CdtGen => "CDT_GEN", Var::Shuf => "SHUF", Var::Corrupt => "CORRUPT", Var::CdtRand => "CDT_RANDPRUNE", Var::CdtFixed => "CDT_FIXED" };
        // sueño A
        let ex: Vec<Example> = s.ds.train.clone();
        let sa = consolidate_v(v, &ex, seed);
        let mut mem = Mem { v, a: if sa.rules.is_empty() && sa.generic.is_none() { None } else { Some(sa) }, b: None, tau: [0.05; N_FAM], nprev: 0, sleeps: 1 };
        for f in 0..N_FAM {
            let rt = mem.active(f);
            let mut rs: Vec<f64> = a_wake.iter().filter(|e| e.fam == f).map(|e| m::rel_err(&mem.one(d, rt, f, e.p, &e.x0), &e.ys[0])).collect();
            rs.sort_by(|x, y| x.total_cmp(y));
            if !rs.is_empty() { mem.tau[f] = (2.0 * rs[((rs.len() as f64 * 0.95) as usize).min(rs.len() - 1)]).max(0.05); }
        }
        // vigilia B con disparo por saturación; RAW guarda la vigilia completa (A luego B) con el slot asignado
        let mut wake_log: Vec<(usize, bool, Ep)> = a_wake.iter().map(|e| (e.fam, false, e.clone())).collect();
        let (mut buf, mut win, mut nov_n, mut trig, mut win_i) = (vec![], std::collections::VecDeque::new(), 0usize, 0usize, 0usize);
        for e in &b_wake {
            let (rt, _r, nov) = mem.route(d, e);
            win.push_back(nov); if win.len() > W { win.pop_front(); }
            if nov { nov_n += 1; buf.push(e.clone()); }
            wake_log.push((e.fam, nov || rt == Route::B, e.clone()));
            let rho = win.iter().filter(|&&x| x).count() as f64 / win.len() as f64;
            let i_ep = win_i; win_i += 1;
            let fire = if v == Var::CdtFixed { let every = (1024 / n_b_sleeps).max(1); (i_ep + 1) % every == 0 && trig + 1 < n_b_sleeps }
                       else { (win.len() == W && rho > RHO && buf.len() >= 64) || buf.len() >= NMAX };
            if fire { mem.sleep_b(&mut buf, seed); trig += 1; }
        }
        mem.sleep_b(&mut buf, seed);
        if v == Var::Cdt { n_b_sleeps = (mem.sleeps - 1).max(1); }
        let numbers = nnum(&mem.a) + nnum(&mem.b);
        let mut rr = Xoshiro256StarStar::seed_from_u64(seed ^ 0x5A5);
        let mut raw = Raw { cap: numbers / 26, seen: 0, items: vec![] };
        for it in wake_log { raw.push(&mut rr, it); }
        for e in &evals {
            let (rt, r, _) = mem.route(d, e);
            let (o1, o4) = rollout(|x| mem.one(d, rt, e.fam, e.p, x), e);
            let slot = match rt { Route::B => 8 + BASE.iter().position(|&b| b == e.fam).unwrap(), _ => e.fam };
            println!("{},{},{},{name},{},{},{r},{slot},{},{}", e.set, e.truefam, e.fam, o1 as u8, o4 as u8, e.p[0], e.p[1]);
            if !stage2 && matches!(v, Var::Cdt | Var::CdtNp | Var::CdtGen) {
                let (o1, o4) = rollout(|x| match rt { Route::Orb => d.step(x, &context_of(e.fam, e.p)),
                    _ => raw.step(e.fam, rt == Route::B, e.p, x).unwrap_or_else(|| d.step(x, &context_of(e.fam, e.p))) }, e);
                println!("{},{},{},RAW_{name},{},{},{r}", e.set, e.truefam, e.fam, o1 as u8, o4 as u8);
            }
        }
        let fpr = a_eval.iter().filter(|e| mem.route(d, e).1 > mem.tau[e.fam]).count() as f64 / a_eval.len() as f64;
        meta.push(format!("{name}: sleeps={} trig={trig} novel_B_wake={nov_n}/1024 fprA={fpr:.4} numbers={numbers} rawcap={} rulesA={} rulesB={} tau={:?}",
            mem.sleeps, raw.cap, mem.a.as_ref().map(|s| s.rules.len()).unwrap_or(0), mem.b.as_ref().map(|s| s.rules.len()).unwrap_or(0), mem.tau.iter().map(|t| (t * 1e4).round() / 1e4).collect::<Vec<_>>()));
    }
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before);
    eprintln!("seed=0x{seed:X} n_A_eval={} n_B_eval={} test_read=never queries_total={} (CDT/replay internos) {}", a_eval.len(), b_eval.len(), q.total(), meta.join(" | "));
}
