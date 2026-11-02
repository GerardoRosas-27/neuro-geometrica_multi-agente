//! BIO-CONSTELACIÓN portada al arnés CONSTELACIÓN v4 (docs/bio_constelacion/PREREG.md).
//! Predictor residual equivariante por punto: q̂(t+1) = q(t) + (W^P + W^F)·φ(q, μ), φ ∈ R^10 (base L2/3 de bio_constelacion/hybrid_brain.rs,
//! con μ = latente Koopman L5 contractivo por punto). W^P lento (flujos A+B del protocolo v4), W^F rápido (se resetea por episodio,
//! se adapta sólo con transiciones OBSERVADAS y causales: la predicción del cuadro t usa W^F ajustado con transiciones < t; sin fuga en h=1).
//! Componente bio (brazo `hyb`): Ley de Dale (W = W^E − W^I, ambos ≥ 1e-4, apply_dale_pair) + STP (depresión de recursos por sinapsis
//! que escala la actualización rápida) + consolidación SWR (W^P += c·W^F al final de cada episodio de entrenamiento).
//! Ablación `hyb_nobio`: mismos φ/W^P/W^F, pesos sin signo, sin STP, sin consolidación.
//!   orbita bio-run --phase dev|confirm|smoke --seed S --arm hyb|hyb_nobio|ref --weights W --frozen F --out DIR
//!   orbita bio-tune --seeds 0xC420,... --weights W   (sólo semillas de ajuste)
use super::*;
use std::sync::Arc;
const D: usize = 10;

#[derive(Clone, Debug)]
pub struct BioCfg { pub bio: bool, pub eta_p: f64, pub eta_f: f64, pub t_f: usize, pub cons: f64, pub stp_u: f64, pub stp_tau: f64, pub rho: f64, pub omega: f64 }
impl BioCfg {
    pub fn from_json(v: &serde_json::Value, bio: bool) -> BioCfg { let g = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("falta {k}"));
        BioCfg { bio, eta_p: g("eta_p"), eta_f: g("eta_f"), t_f: g("t_f") as usize, cons: if bio { g("cons") } else { 0.0 }, stp_u: g("stp_u"), stp_tau: g("stp_tau"), rho: g("rho"), omega: g("omega") } }
}
#[derive(Clone)]
pub struct Bio { pub c: BioCfg, pe: Vec<f64>, pi: Vec<f64>, fe: Vec<f64>, fi: Vec<f64>, res: Vec<f64>, pub nan: bool, pub dale_viol: usize }
const WMIN: f64 = 1e-4;
fn dale(we: &mut f64, wi: &mut f64, dw: f64) { if dw >= 0.0 { *we += dw; } else { *wi -= dw; } let m = we.min(*wi) - WMIN; if m > 0.0 { *we -= m; *wi -= m; } *we = we.max(WMIN); *wi = wi.max(WMIN); }
impl Bio {
    pub fn new(c: BioCfg) -> Bio { let z = vec![if c.bio { WMIN } else { 0.0 }; 2 * D]; Bio { c, pe: z.clone(), pi: z.clone(), fe: z.clone(), fi: z.clone(), res: vec![1.0; 2 * D], nan: false, dale_viol: 0 } }
    fn begin(&mut self) { let v = if self.c.bio { WMIN } else { 0.0 }; for w in self.fe.iter_mut().chain(self.fi.iter_mut()) { *w = v; } for r in self.res.iter_mut() { *r = 1.0; } }
    fn w(&self, i: usize) -> f64 { (self.pe[i] - self.pi[i]) + (self.fe[i] - self.fi[i]) }
    fn phi(q: &[f64], mu: &[f64]) -> [f64; D] { let (x, y) = (q[0], q[1]);
        [x, y, 1.0, (x * x + y * y).sqrt(), (2.0 * x).tanh(), (2.0 * y).tanh(), (2.0 * x).sin(), (2.0 * y).sin(), mu[0], mu[1]] }
    fn koop(&self, mu: &mut [f64], q: &[f64]) { let (s, co) = self.c.omega.sin_cos(); let r = self.c.rho; let (a, b) = (r * (co * mu[0] - s * mu[1]), r * (s * mu[0] + co * mu[1])); mu[0] = a + (1.0 - r) * q[0]; mu[1] = b + (1.0 - r) * q[1]; }
    fn pred_pt(&self, q: &[f64], mu: &[f64]) -> [f64; 2] { let f = Self::phi(q, mu); let mut o = [q[0], q[1]]; for c in 0..2 { for j in 0..D { o[c] += self.w(c * D + j) * f[j]; } } o }
    fn pred(&self, x: &[f64], mus: &[[f64; 2]]) -> Vec<f64> { (0..x.len() / 2).flat_map(|p| self.pred_pt(&x[2 * p..2 * p + 2], &mus[p])).collect() }
    /// una actualización NLMS sobre una transición (todos los puntos); slow: W^P, si no W^F
    fn nlms(&mut self, x: &[f64], y: &[f64], mus: &[[f64; 2]], slow: bool, eta: f64) {
        let np = x.len() / 2; let mut g = vec![0.0; 2 * D]; let mut nn = 1e-8;
        for p in 0..np { let f = Self::phi(&x[2 * p..2 * p + 2], &mus[p]); let o = self.pred_pt(&x[2 * p..2 * p + 2], &mus[p]);
            for c in 0..2 { let e = y[2 * p + c] - o[c]; for j in 0..D { g[c * D + j] += e * f[j]; } } nn += f.iter().map(|v| v * v).sum::<f64>(); }
        for i in 0..2 * D { let mut dw = eta * g[i] / nn; if !slow && self.c.bio { dw *= self.res[i]; }
            let (we, wi) = if slow { (&mut self.pe[i], &mut self.pi[i]) } else { (&mut self.fe[i], &mut self.fi[i]) };
            if self.c.bio { dale(we, wi, dw); } else { *we += dw; } }
        if !slow && self.c.bio { for i in 0..2 * D { let u = (g[i].abs() / nn.sqrt()).min(1.0); self.res[i] += (1.0 - self.res[i]) / self.c.stp_tau - self.c.stp_u * u * self.res[i]; self.res[i] = self.res[i].clamp(0.05, 1.0); } }
        if !self.pe.iter().chain(&self.fe).all(|v| v.is_finite()) { self.nan = true; }
    }
    fn mus_after(&self, e: &Ep, t: usize) -> Vec<Vec<[f64; 2]>> { // μ antes de procesar cada cuadro 0..=t
        let np = e.x0.len() / 2; let mut m = vec![[0.0; 2]; np]; let mut out = vec![];
        for s in 0..=t { out.push(m.clone()); let x = e.frame(s); for p in 0..np { let mut mm = m[p]; self.koop(&mut mm, &x[2 * p..2 * p + 2]); m[p] = mm; } } out }
    /// episodio: train = 16 transiciones (NLMS lento + rápido, consolidación). eval: observa 0..=k, preds[h−1] (h≤k filtrado causal, h>k rollout)
    pub fn episode(&mut self, e: &Ep, train: bool, k: usize) -> Vec<Vec<f64>> {
        self.begin(); let last = if train { HORIZON } else { k }; let mus = self.mus_after(e, last); let mut preds = vec![vec![]; HORIZON];
        for t in 1..=last { let (x, y) = (e.frame(t - 1), e.frame(t));
            if !train && t <= k { preds[t - 1] = self.pred(x, &mus[t - 1]); } // predicción ANTES de ver y(t)
            if train { self.nlms(x, y, &mus[t - 1], true, self.c.eta_p); }
            // rápido: T_F iteraciones sobre todas las transiciones observadas hasta t (≤ t), como v4
            let it = (self.c.t_f / t).max(1); for _ in 0..it { for s in 1..=t { self.nlms(e.frame(s - 1), e.frame(s), &mus[s - 1], false, self.c.eta_f); } }
            if self.nan { break; } }
        if train { if self.c.cons > 0.0 { for i in 0..2 * D { let dw = self.c.cons * (self.fe[i] - self.fi[i]); let (we, wi) = (&mut self.pe[i], &mut self.pi[i]); dale(we, wi, dw); } } return preds; }
        // rollout reinyectado desde x_k con W congelados
        let np = e.x0.len() / 2; let mut m = mus[k].clone(); let mut z = e.frame(k).to_vec();
        for p in 0..np { let mut mm = m[p]; self.koop(&mut mm, &z[2 * p..2 * p + 2]); m[p] = mm; }
        for h in (k + 1)..=HORIZON { let p = self.pred(&z, &m); preds[h - 1] = p.clone(); if !fin(&p) { break; } z = p; for q in 0..np { let mut mm = m[q]; self.koop(&mut mm, &z[2 * q..2 * q + 2]); m[q] = mm; } }
        preds }
    pub fn audit(&self) -> usize { if !self.c.bio { return 0; } self.pe.iter().chain(&self.pi).chain(&self.fe).chain(&self.fi).filter(|&&v| v < 0.0).count() }
    pub fn hash(&self) -> u64 { let mut h: u64 = 0xcbf29ce484222325; for v in self.pe.iter().chain(&self.pi) { h ^= v.to_bits(); h = h.wrapping_mul(0x100000001b3); } h }
}
fn h16(b: &mut Bio, e: &Ep, k: usize) -> f64 { let p = b.episode(e, false, k); clamp_err(&p[HORIZON - 1], &e.ys[HORIZON - 1]) }
pub fn train(b: &mut Bio, seed: u64, na: usize) { let sa = gen_known_stream(seed, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()); let sb = gen_stream(seed, N_NEW, N_KNOWN_B);
    for e in sa.iter().chain(&sb) { b.episode(e, true, K); if b.nan { break; } } }
pub fn eval(b: &mut Bio, held: &[Ep], ex: &[Example]) -> serde_json::Value {
    let mut r = serde_json::Map::new();
    for (g, known) in [("known", true), ("new", false)] { let eps: Vec<&Ep> = held.iter().filter(|e| (e.fam < N_FAM) == known).collect();
        let (mut e16, mut h1) = (vec![], vec![]);
        for e in &eps { let p = b.episode(e, false, K); e16.push(clamp_err(&p[HORIZON - 1], &e.ys[HORIZON - 1])); h1.push(log_rel(clamp_err(&p[0], e.frame(1)))); }
        r.insert(format!("{g}_acc16"), json!(e16.iter().filter(|&&x| x < mt::REL_ERR_OK).count() as f64 / e16.len() as f64));
        r.insert(format!("{g}_med_log"), json!(median(e16.iter().map(|&x| log_rel(x)).collect())));
        r.insert(format!("{g}_h1_med_log"), json!(median(h1)));
        for k in [2usize, 8] { r.insert(format!("{g}_med_log_k{k}"), json!(median(eps.iter().map(|e| log_rel(h16(b, e, k))).collect()))); } }
    let trs: Vec<(Vec<Vec<f64>>, &Example)> = ex.iter().map(|e| { let ys: Vec<Vec<f64>> = (1..=HORIZON).map(|h| target_h(e, h)).collect();
        let ep = Ep { fam: e.family_id, p: [0.0; 2], x0: e.input.clone(), ys }; (b.episode(&ep, false, K), e) }).collect();
    r.insert("e33".into(), e33_score(&trs));
    serde_json::Value::Object(r)
}
/// persistencia: x̂16 = x_K
pub fn persist_held(held: &[Ep]) -> serde_json::Value { let mut r = serde_json::Map::new();
    for (g, known) in [("known", true), ("new", false)] { let errs: Vec<f64> = held.iter().filter(|e| (e.fam < N_FAM) == known).map(|e| clamp_err(e.frame(K), &e.ys[HORIZON - 1])).collect();
        r.insert(format!("{g}_acc16"), json!(errs.iter().filter(|&&x| x < mt::REL_ERR_OK).count() as f64 / errs.len() as f64));
        r.insert(format!("{g}_med_log"), json!(median(errs.iter().map(|&x| log_rel(x)).collect()))); }
    serde_json::Value::Object(r) }

pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); let phase = arg_of(a, "--phase").unwrap(); let arm = arg_of(a, "--arm").unwrap();
    match phase.as_str() { "dev" => assert!((0xA400..=0xA40F).contains(&seed)), "confirm" => assert!((0xB400..=0xB40F).contains(&seed)), _ => assert!(phase == "smoke" && ((0xC400..=0xC4FF).contains(&seed) || (0xD400..=0xD4FF).contains(&seed))) }
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}_{arm}.done")); if done.exists() { eprintln!("skip"); return; }
    let t0 = std::time::Instant::now(); let before = cdt_rqm_epr::v4_provenance::snapshot();
    let fz_txt = std::fs::read_to_string(arg_of(a, "--frozen").unwrap()).unwrap(); let fz: serde_json::Value = serde_json::from_str(&fz_txt).unwrap();
    let held = gen_heldout(seed, HPF); let (ex, sha) = e33_examples(seed);
    let r = match arm.as_str() {
        "hyb" | "hyb_nobio" => { let mut b = Bio::new(BioCfg::from_json(&fz["hyb"], arm == "hyb")); train(&mut b, seed, N_A); let ev = eval(&mut b, &held, &ex);
            json!({"held": ev, "nan": b.nan, "dale_negative_weights": b.audit(), "hash": format!("{:016x}", b.hash()), "cfg": format!("{:?}", b.c)}) }
        "ref" => { let d = Arc::new(Orbita::from_json(&std::fs::read_to_string(arg_of(a, "--weights").unwrap()).unwrap()));
            json!({"orbita": orbita_held(&d, &held), "persist": persist_held(&held), "orbita_e33": e33_orbita(&d, &ex)}) }
        _ => panic!("brazo desconocido {arm}") };
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(); assert_eq!(q, 0, "consultas TEST ≠ 0");
    let mut r = r; r["seed"] = json!(format!("0x{seed:X}")); r["phase"] = json!(phase); r["arm"] = json!(arm); r["test_queries"] = json!(q); r["test_sha256"] = json!(sha); r["git_sha"] = json!(git_sha()); r["wall_s"] = json!(t0.elapsed().as_secs_f64());
    std::fs::write(out.join(format!("metrics_0x{seed:X}_{arm}.json")), serde_json::to_string(&r).unwrap()).unwrap(); std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} {arm} ok {:.0}s", t0.elapsed().as_secs_f64());
}
/// ajuste (sólo 0xC4xx/0xD4xx; criterio: media de new_med_log sobre held-out de ajuste; sin TEST/E33)
pub fn tune(a: &[String]) {
    let seeds: Vec<u64> = arg_of(a, "--seeds").unwrap().split(',').map(parse_seed).collect();
    assert!(seeds.iter().all(|&s| (0xC400..=0xC4FF).contains(&s) || (0xD400..=0xD4FF).contains(&s)));
    let base: serde_json::Value = serde_json::from_str(&arg_of(a, "--cfg").unwrap()).unwrap(); let na: usize = arg_of(a, "--na").map(|v| v.parse().unwrap()).unwrap_or(N_A);
    for bio in [true, false] { let mut v = vec![]; let mut kn = vec![];
        for &s in &seeds { let mut b = Bio::new(BioCfg::from_json(&base, bio)); train(&mut b, s, na); let held = gen_heldout(s, HPF);
            v.push(median(held.iter().filter(|e| e.fam >= N_FAM).map(|e| log_rel(h16(&mut b, e, K))).collect()));
            kn.push(median(held.iter().filter(|e| e.fam < N_FAM).map(|e| log_rel(h16(&mut b, e, K))).collect())); }
        println!("{}", json!({"cfg": base, "bio": bio, "new_med_log": v, "mean_new": mean(&v), "known_med_log": kn, "mean_known": mean(&kn)})); }
}
