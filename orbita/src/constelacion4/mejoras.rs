//! CONSTELACIÓN v4 + M1/M2/M3 (docs/constelacion_v4_mejoras/PREREG.md). Drivers: paso0, calib, tune, meta (ES), run.
use super::*;
use rayon::prelude::*;
use std::sync::Arc;

const XOR_CALIB: u64 = 0xCA11_B0B0; const XOR_META_T: u64 = 0x3E7A_0001; const XOR_META_V: u64 = 0x3E7A_0002;
fn seeds_of(a: &[String], k: &str) -> Vec<u64> { arg_of(a, k).unwrap().split(',').map(parse_seed).collect() }
fn tune_seed(s: u64) -> bool { (0xC400..=0xC4FF).contains(&s) || (0xD400..=0xD4FF).contains(&s) }
fn load_orb(a: &[String]) -> Arc<Orbita> { Arc::new(Orbita::from_json(&std::fs::read_to_string(arg_of(a, "--weights").unwrap()).unwrap())) }
fn fnv(it: impl Iterator<Item = u64>) -> u64 { let mut h: u64 = 0xcbf29ce484222325; for v in it { h ^= v; h = h.wrapping_mul(0x100000001b3); } h }
fn f(v: &serde_json::Value, k: &str) -> f64 { v[k].as_f64().unwrap_or_else(|| panic!("falta {k}")) }

/// límites de m para NLMS con momento: η_F·m/(1−β) < 2
pub fn m_bounds(beta: f64, eta_f: f64) -> (f64, f64) { let mx = (2.0 * (1.0 - beta) / eta_f).min(2.0); (0.25f64.min(mx), mx) }
pub fn set_m1(c: &mut Cfg, g: f64, b: f64, l: f64, mu: f64, sd: f64) { c.m1_gamma = g; c.m1_beta = b; c.m1_lam = l; let (lo, hi) = m_bounds(b, c.eta_f); c.m1_mmin = lo; c.m1_mmax = hi; c.m1_mu = mu; c.m1_sd = sd; }
pub fn set_m3(c: &mut Cfg, h: usize, ev: f64, eu: f64, s3: usize) { c.m3_h = h; c.m3_eta_v = ev; c.m3_eta_u = eu; c.m3_s3 = s3; }
/// aplica la config a una red ya entrenada (clon) sin tocar los pesos lentos
fn with_cfg(base: &Net, c: &Cfg, seed: u64) -> Net { let mut n = base.clone(); n.c = c.clone(); n.m3h = c.m3_h;
    if c.m3_h > 0 { let fresh = Net::new(c, seed); n.fu0 = fresh.fu0; n.fu = n.fu0.clone(); n.fb = vec![0.0; c.m3_h]; n.fv = vec![0.0; 2 * c.m3_h]; } else { n.fu0 = vec![]; n.fu = vec![]; n.fb = vec![]; n.fv = vec![]; } n }

/// configuración congelada por brazo (frozen.json)
pub fn arm_cfg(arm: &str, fz: &serde_json::Value) -> Cfg {
    let mut c = Cfg::v4(); let m1 = &fz["m1"]; let m3 = &fz["m3"];
    let a1 = |c: &mut Cfg| set_m1(c, f(m1, "gamma"), f(m1, "beta"), f(m1, "lam"), f(m1, "mu"), f(m1, "sd"));
    let a3 = |c: &mut Cfg| set_m3(c, m3["h"].as_u64().unwrap() as usize, f(m3, "eta_v"), f(m3, "eta_u"), m3["s3"].as_u64().unwrap() as usize);
    let th = || -> Option<[f64; 6]> { if fz["m2"]["accepted"].as_bool() == Some(true) { let v: Vec<f64> = fz["m2"]["theta"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect(); Some([v[0], v[1], v[2], v[3], v[4], v[5]]) } else { None } };
    match arm {
        "v4" => {}
        "m1" => a1(&mut c),
        "m1_ctrl" => { a1(&mut c); c.m1_gamma = 0.0; c.m1_const = f(m1, "m_mean"); }
        "m2" => { c.m2 = Some(th().expect("M2 no aceptado en meta-val: no va a DEV")); c.m1_mu = f(m1, "mu"); c.m1_sd = f(m1, "sd"); }
        "m3" => a3(&mut c),
        "m3_rf" => { a3(&mut c); c.m3_eta_u = 0.0; }
        "m3_orb" => { a3(&mut c); c.m3_orb = true; }
        "all" => { a1(&mut c); a3(&mut c); c.m2 = th(); }
        _ => panic!("brazo desconocido {arm}"),
    }
    c
}
fn mk_net(c: &Cfg, seed: u64, d: &Arc<Orbita>) -> Net { let mut n = Net::new(c, seed); n.orb = Some(d.clone()); n }

/// h16 por episodio con k observados
fn h16s(net: &mut Net, eps: &[Ep], k: usize) -> Vec<f64> { eps.iter().map(|e| run_episode(net, e, false, 0.0, false, k).h16_err).collect() }
fn medlog(v: &[f64]) -> f64 { median(v.iter().map(|&x| log_rel(x)).collect()) }

/// evaluación completa de un brazo entrenado
fn eval_arm(net: &mut Net, held: &[Ep], d: &Orbita, ex: &[Example]) -> serde_json::Value {
    let mut r = serde_json::Map::new();
    for (g, known) in [("known", true), ("new", false)] {
        let eps: Vec<Ep> = held.iter().filter(|e| (e.fam < N_FAM) == known).cloned().collect();
        net.m_log.clear(); let (mut e16, mut h1, mut p1) = (vec![], vec![], vec![]);
        for e in &eps { let o = run_episode(net, e, false, 0.0, false, K); e16.push(o.h16_err); h1.push(log_rel(clamp_err(&o.preds[K], e.frame(K + 1)))); p1.push(log_rel(clamp_err(e.frame(K), e.frame(K + 1)))); }
        r.insert(format!("{g}_acc16"), json!(e16.iter().filter(|&&x| x < mt::REL_ERR_OK).count() as f64 / e16.len() as f64));
        r.insert(format!("{g}_med_log"), json!(medlog(&e16)));
        r.insert(format!("{g}_m_mean"), json!(if net.m_log.is_empty() { 1.0 } else { mean(&net.m_log) }));
        r.insert(format!("{g}_h1_med_log"), json!(median(h1))); r.insert(format!("{g}_h1_persist_med_log"), json!(median(p1)));
        for k in [2usize, 8] { r.insert(format!("{g}_med_log_k{k}"), json!(medlog(&h16s(net, &eps, k)))); }
    }
    let (e33, e33raw) = e33_net2(net, d, ex);
    r.insert("e33".into(), e33); r.insert("e33_v3head".into(), e33raw);
    serde_json::Value::Object(r)
}

/// F2 por (semilla, brazo): entrenamiento desde cero en flujos A+B con la regla del brazo
pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); let phase = arg_of(a, "--phase").unwrap(); let arm = arg_of(a, "--arm").unwrap();
    match phase.as_str() { "dev" => assert!((0xA400..=0xA40F).contains(&seed)), "confirm" => assert!((0xB400..=0xB40F).contains(&seed)), _ => assert!(phase == "smoke" && tune_seed(seed)) }
    let fz_path = arg_of(a, "--frozen").unwrap(); let fz_txt = std::fs::read_to_string(&fz_path).unwrap(); let fz: serde_json::Value = serde_json::from_str(&fz_txt).unwrap();
    let wpath = arg_of(a, "--weights").unwrap(); let d = Arc::new(Orbita::from_json(&std::fs::read_to_string(&wpath).unwrap())); let raw0 = d.raw.clone();
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}_{arm}.done")); if done.exists() { eprintln!("skip"); return; }
    let t0 = std::time::Instant::now(); let before = cdt_rqm_epr::v4_provenance::snapshot();
    let c = arm_cfg(&arm, &fz);
    let na: usize = if phase == "smoke" { arg_of(a, "--na").map(|v| v.parse().unwrap()).unwrap_or(N_A) } else { N_A };
    let held = gen_heldout(seed, HPF); let held_known: Vec<Ep> = held.iter().filter(|e| e.fam < N_FAM).cloned().collect();
    let sa = gen_known_stream(seed, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()); let sb = gen_stream(seed, N_NEW, N_KNOWN_B);
    let (ex, sha) = e33_examples(seed);
    let mut net = mk_net(&c, seed, &d); let mut cdt = Cdt::new(seed, &c);
    run_stream(&mut net, &mut cdt, &sa, seed); let pre = eval_held(&mut net, &held_known);
    let (_, h16b, fb) = run_stream(&mut net, &mut cdt, &sb, seed);
    let train_m = if net.m_log.is_empty() { 1.0 } else { mean(&net.m_log) };
    let ev = eval_arm(&mut net, &held, &d, &ex); let col = collapse(&mut net, &held);
    let auc_new = mean(&h16b.iter().zip(&fb).filter(|(_, &f)| f >= N_FAM).map(|(v, _)| *v).collect::<Vec<_>>());
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    assert_eq!(d.raw, raw0, "ÓRBITA modificada"); assert_eq!(q, 0, "consultas TEST ≠ 0");
    let r = json!({"seed": format!("0x{seed:X}"), "phase": phase, "arm": arm, "weights": wpath, "frozen": fz, "frozen_fnv": format!("{:016x}", fnv(fz_txt.bytes().map(|b| b as u64))),
        "held": ev, "pre_known": pre, "collapse": col, "auc_new_preq": auc_new, "nan": net.nan, "hash": format!("{:016x}", net.hash()), "sleeps": cdt.log.len(), "eta_scale": net.eta_scale,
        "train_m_mean": train_m, "orb_calls": net.orb_calls, "test_queries": q, "test_sha256": sha, "git_sha": git_sha(), "cfg": c.to_json(), "wall_s": t0.elapsed().as_secs_f64()});
    std::fs::write(out.join(format!("metrics_0x{seed:X}_{arm}.json")), serde_json::to_string(&r).unwrap()).unwrap(); std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} {arm} ok {:.0}s", t0.elapsed().as_secs_f64());
}

/// surprise log S por cuadro (t=1..=16) de un episodio con contexto causal
pub fn surprises(d: &Orbita, e: &Ep) -> Vec<f64> { let ctx = reid_prefixes(d, &e.x0, &e.ys, K, [0.0; 2]); let np = e.x0.len() / 2;
    (1..=HORIZON).map(|t| { let o = d.step(e.frame(t - 1), &ctx[(t - 1).min(K)]); let x = e.frame(t); (0..np).map(|p| (x[2 * p] - o[2 * p]).powi(2) + (x[2 * p + 1] - o[2 * p + 1]).powi(2)).sum::<f64>() / np as f64 }).collect() }

/// calibración μ_S, σ_S (mediana, MAD·1.4826 de ln S) en conocidas de semillas de ajuste; media de m por (γ, β)
pub fn calib(a: &[String]) {
    let seeds = seeds_of(a, "--seeds"); assert!(seeds.iter().all(|&s| tune_seed(s))); let d = load_orb(a);
    let eps: Vec<Ep> = seeds.iter().flat_map(|&s| gen_known_stream(s, XOR_CALIB, 256, &(0..N_FAM).collect::<Vec<_>>())).collect();
    let ls: Vec<f64> = eps.par_iter().flat_map(|e| surprises(&d, e).into_iter().map(|s| s.max(1e-300).ln()).collect::<Vec<_>>()).collect();
    let mu = median(ls.clone()); let sd = median(ls.iter().map(|v| (v - mu).abs()).collect()) * 1.4826;
    let mut mm = serde_json::Map::new();
    for g in [0.25, 0.5, 1.0] { for b in [0.0, 0.5, 0.9] { let (lo, hi) = m_bounds(b, V3_ETA_F);
        mm.insert(format!("g{g}_b{b}"), json!(mean(&ls.iter().map(|v| (1.0 + g * ((v - mu) / sd).tanh()).clamp(lo, hi)).collect::<Vec<_>>()))); } }
    println!("{}", json!({"seeds": seeds.iter().map(|s| format!("0x{s:X}")).collect::<Vec<_>>(), "n": ls.len(), "mu": mu, "sd": sd, "m_mean": mm}));
}

/// grilla de ajuste: (nombre, cfg)
fn tune_grid(mu: f64, sd: f64) -> Vec<(String, Cfg)> {
    let mut v = vec![("v4".to_string(), Cfg::v4())];
    for g in [0.25, 0.5, 1.0] { for b in [0.0, 0.5, 0.9] { for l in [0.0, 0.05, 0.2] { let mut c = Cfg::v4(); set_m1(&mut c, g, b, l, mu, sd); v.push((format!("m1_g{g}_b{b}_l{l}"), c)); } } }
    let mut m3 = vec![]; for h in [16, 32, 64] { for eu in [0.0, 0.05] { for s3 in [8, 32] { m3.push((h, 0.5, eu, s3)); } } }
    m3.push((32, 0.1, 0.05, 32)); m3.push((32, 0.5, 0.01, 32)); m3.push((32, 0.5, 0.05, 128));
    for (h, ev, eu, s3) in m3 { let mut c = Cfg::v4(); set_m3(&mut c, h, ev, eu, s3); v.push((format!("m3_h{h}_ev{ev}_eu{eu}_s{s3}"), c)); }
    v
}
/// ajuste M1/M3 (sólo semillas 0xC4xx/0xD4xx): red v4 entrenada en el flujo A; cada config se evalúa sólo en H conocidas, k∈{2,8} (criterio) y k=4 (info)
pub fn tune(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); assert!(tune_seed(seed)); let d = load_orb(a);
    let cal: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(arg_of(a, "--calib").unwrap()).unwrap()).unwrap();
    let t0 = std::time::Instant::now(); let na: usize = arg_of(a, "--na").map(|v| v.parse().unwrap()).unwrap_or(N_A);
    let c4 = Cfg::v4(); let mut base = mk_net(&c4, seed, &d); let mut cdt = Cdt::new(seed, &c4);
    run_stream(&mut base, &mut cdt, &gen_known_stream(seed, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()), seed);
    let t_train = t0.elapsed().as_secs_f64();
    let held: Vec<Ep> = gen_heldout(seed, HPF).into_iter().filter(|e| e.fam < N_FAM).collect(); assert!(held.iter().all(|e| e.fam < N_FAM));
    let only = arg_of(a, "--only");
    let grid: Vec<(String, Cfg)> = tune_grid(f(&cal, "mu"), f(&cal, "sd")).into_iter().filter(|(n, _)| only.as_ref().map_or(true, |o| n.starts_with(o.as_str()) || n == "v4")).collect();
    let res: Vec<(String, serde_json::Value)> = grid.par_iter().map(|(name, c)| { let ta = std::time::Instant::now(); let mut n = with_cfg(&base, c, seed);
        let k2 = medlog(&h16s(&mut n, &held, 2)); let k8 = medlog(&h16s(&mut n, &held, 8)); let k4 = medlog(&h16s(&mut n, &held, K));
        (name.clone(), json!({"k2": k2, "k8": k8, "k4": k4, "crit": (k2 + k8) / 2.0, "nan": n.nan, "m_mean": if n.m_log.is_empty() { 1.0 } else { mean(&n.m_log) }, "wall_s": ta.elapsed().as_secs_f64()})) }).collect();
    let mut m = serde_json::Map::new(); for (k, v) in res { m.insert(k, v); }
    println!("{}", json!({"seed": format!("0x{seed:X}"), "train_s": t_train, "na": na, "calib": cal, "res": m, "nan_base": base.nan}));
}

/// M2: ES (OpenAI-ES antitético, rank shaping) sobre θ (6 coef. globales), inicio = NLMS; meta-train fam 0–5, meta-val fam 6–7; semillas de ajuste
pub fn meta(a: &[String]) {
    let seeds = seeds_of(a, "--seeds"); assert!(seeds.iter().all(|&s| tune_seed(s))); let d = load_orb(a); let raw0 = d.raw.clone();
    let cal: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(arg_of(a, "--calib").unwrap()).unwrap()).unwrap();
    let gens: usize = arg_of(a, "--gens").map(|v| v.parse().unwrap()).unwrap_or(50); let pop = 16usize; let (sig, lr, l2) = (0.05, 0.02, 0.1);
    let ept: usize = arg_of(a, "--ept").map(|v| v.parse().unwrap()).unwrap_or(32); let epv: usize = arg_of(a, "--epv").map(|v| v.parse().unwrap()).unwrap_or(16);
    let na: usize = arg_of(a, "--na").map(|v| v.parse().unwrap()).unwrap_or(N_A);
    let before = cdt_rqm_epr::v4_provenance::snapshot(); let t0 = std::time::Instant::now();
    let mut c0 = Cfg::v4(); c0.m1_mu = f(&cal, "mu"); c0.m1_sd = f(&cal, "sd");
    let bases: Vec<Net> = seeds.par_iter().map(|&s| { let mut n = mk_net(&Cfg::v4(), s, &d); let mut cdt = Cdt::new(s, &n.c.clone()); run_stream(&mut n, &mut cdt, &gen_known_stream(s, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()), s); n }).collect();
    let t_train = t0.elapsed().as_secs_f64();
    let mk = |xor: u64, n: usize, fams: &[usize]| -> Vec<(usize, Ep, usize)> { let mut v = vec![]; for (i, &s) in seeds.iter().enumerate() { for (j, e) in gen_known_stream(s, xor, n, fams).into_iter().enumerate() { v.push((i, e, [2usize, 4, 8][j % 3])); } } v };
    let tr = mk(XOR_META_T, ept, &[0, 1, 2, 3, 4, 5]); let va = mk(XOR_META_V, epv, &[6, 7]);
    assert!(tr.iter().all(|x| x.1.fam < 6)); assert!(va.iter().all(|x| x.1.fam == 6 || x.1.fam == 7));
    let fit = |th: &[f64; 6], set: &[(usize, Ep, usize)]| -> f64 { let mut c = c0.clone(); c.m2 = Some(*th);
        let r: Vec<f64> = set.par_iter().map(|(i, e, k)| { let mut n = with_cfg(&bases[*i], &c, seeds[*i]); let o = run_episode(&mut n, e, false, 0.0, false, *k); if n.nan { f64::NEG_INFINITY } else { -log_rel(o.h16_err) } }).collect();
        if r.iter().any(|v| !v.is_finite()) { f64::NEG_INFINITY } else { mean(&r) } };
    let proj = |t: &mut [f64; 6]| { for v in t.iter_mut() { *v = v.clamp(-2.0, 2.0); } t[0] = t[0].clamp(1e-3, 1.999); t[5] = t[5].clamp(0.0, 0.5); };
    let th0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; let mut th = th0;
    let tg = std::time::Instant::now(); let v0 = fit(&th0, &va); let f0 = fit(&th0, &tr); let t_eval = tg.elapsed().as_secs_f64();
    let mut best = (v0, th0, 0usize); let mut log = vec![json!({"gen": 0, "theta": th0, "val": v0, "train": f0})];
    let mut rng = Xoshiro256StarStar::seed_from_u64(0x3E7A_5EED);
    for g in 1..=gens {
        let eps: Vec<[f64; 6]> = (0..pop / 2).map(|_| { let mut e = [0.0; 6]; for v in e.iter_mut() { *v = gauss(&mut rng); } e }).collect();
        let mut cands = vec![]; for e in &eps { for s in [1.0, -1.0] { let mut t = th; for q in 0..6 { t[q] += s * sig * e[q]; } proj(&mut t); cands.push((t, s, *e)); } }
        let fs: Vec<f64> = cands.iter().map(|(t, _, _)| fit(t, &tr)).collect();
        let mut idx: Vec<usize> = (0..pop).collect(); idx.sort_by(|&x, &y| fs[x].total_cmp(&fs[y]));
        let mut rk = vec![0.0; pop]; for (r, &i) in idx.iter().enumerate() { rk[i] = r as f64 / (pop - 1) as f64 - 0.5; }
        let mut gr = [0.0; 6]; for (j, (_, s, e)) in cands.iter().enumerate() { for q in 0..6 { gr[q] += rk[j] * s * e[q]; } }
        for q in 0..6 { th[q] += lr / (pop as f64 * sig) * gr[q] - lr * l2 * (th[q] - th0[q]); } proj(&mut th);
        let v = fit(&th, &va); if v > best.0 { best = (v, th, g); }
        eprintln!("gen {g} val {v:.4} best {:.4} θ {:?} {:.0}s", best.0, th, t0.elapsed().as_secs_f64());
        log.push(json!({"gen": g, "theta": th, "val": v, "train_pop_mean": mean(&fs.iter().cloned().filter(|x| x.is_finite()).collect::<Vec<_>>())}));
    }
    let gain = best.0 - v0; let accepted = gain > 0.02;
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(); assert_eq!(q, 0); assert_eq!(d.raw, raw0);
    println!("{}", json!({"seeds": seeds.iter().map(|s| format!("0x{s:X}")).collect::<Vec<_>>(), "gens": gens, "pop": pop, "sigma": sig, "lr": lr, "l2": l2, "ept": ept, "epv": epv,
        "n_train_eps": tr.len(), "n_val_eps": va.len(), "val_theta0": v0, "train_theta0": f0, "best_val": best.0, "best_gen": best.2, "best_theta": best.1, "gain_val": gain, "accepted": accepted,
        "theta_frozen": if accepted { best.1 } else { th0 }, "test_queries": q, "train_s": t_train, "fitness_eval_s": t_eval / 2.0, "wall_s": t0.elapsed().as_secs_f64(), "log": log}));
}

/// Paso 0 de las mejoras
fn trace_cfg(c: &Cfg, seed: u64, d: &Arc<Orbita>, n_stream: usize) -> (u64, usize, bool, f64) {
    let t0 = std::time::Instant::now(); let mut net = mk_net(c, seed, d); let mut cdt = Cdt::new(seed, c);
    let (h1, h16, _) = run_stream(&mut net, &mut cdt, &gen_known_stream(seed, XOR_KNOWN, n_stream, &(0..N_FAM).collect::<Vec<_>>()), seed);
    let held = gen_heldout(seed, 4); let mut bits: Vec<u64> = h1.iter().chain(&h16).map(|v| v.to_bits()).collect();
    for e in &held { let o = run_episode(&mut net, e, false, 0.0, false, K); bits.extend(o.preds.iter().flatten().map(|v| v.to_bits())); }
    (fnv(bits.into_iter().chain(std::iter::once(net.hash()))), net.orb_calls, net.nan, t0.elapsed().as_secs_f64())
}
pub fn paso0(a: &[String]) {
    let seeds = seeds_of(a, "--seeds"); assert!(seeds.iter().all(|&s| tune_seed(s))); let d = load_orb(a); let raw0 = d.raw.clone();
    let before = cdt_rqm_epr::v4_provenance::snapshot(); let ns: usize = arg_of(a, "--n").map(|v| v.parse().unwrap()).unwrap_or(48);
    let mut res = serde_json::Map::new();
    // (a) bit a bit con cada mejora apagada, forzando la ruta general
    let mut eq = vec![];
    for &s in seeds.iter().take(2) {
        let (h0, _, _, w0) = trace_cfg(&Cfg::v4(), s, &d, ns);
        let mut cases: Vec<(&str, Cfg)> = vec![];
        let mut c = Cfg::v4(); c.gen_path = true; cases.push(("M1a_gamma0_beta0_lam0_genpath", c));
        let mut c = Cfg::v4(); c.m2 = Some([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); cases.push(("M2a_theta0", c));
        let mut c = Cfg::v4(); c.gen_path = true; c.m3_h = 0; cases.push(("M3a_H0", c));
        let mut c = Cfg::v4(); set_m3(&mut c, 32, 0.0, 0.0, 32); cases.push(("M3b_H32_eta0", c));
        for (n, c) in cases { let (h, calls, nan, w) = trace_cfg(&c, s, &d, ns); eq.push(json!({"seed": format!("0x{s:X}"), "case": n, "hash": format!("{h:016x}"), "v4_hash": format!("{h0:016x}"), "pass": h == h0 && !nan, "orb_calls": calls, "wall_s": w, "v4_wall_s": w0})); }
    }
    let ok_a = eq.iter().all(|v| v["pass"] == true); res.insert("bit_a_bit".into(), json!({"cases": eq, "pass": ok_a}));
    // (b) γ=0 ⇒ ÓRBITA no se llama (aunque esté disponible)
    let (_, calls, _, _) = trace_cfg(&Cfg::v4(), seeds[0], &d, ns); let (_, calls1, _, _) = { let mut c = Cfg::v4(); set_m1(&mut c, 0.5, 0.5, 0.05, 0.0, 1.0); trace_cfg(&c, seeds[0], &d, 8) };
    res.insert("M1b_orb_calls".into(), json!({"gamma0": calls, "gamma0.5_n8": calls1, "pass": calls == 0 && calls1 > 0}));
    // (c) causalidad: reid_prefixes ≡ reid(t) y perturbar el futuro no cambia contextos ni S hasta t
    let eps = gen_heldout(seeds[0], 4); let (mut bad_r, mut bad_c, mut n_r) = (0, 0, 0);
    for e in &eps { let ctx = reid_prefixes(&d, &e.x0, &e.ys, K, [0.0; 2]);
        for t in 1..=K { let (g, p, _) = reid(&d, &e.x0, &e.ys, t, [0.0; 2]); n_r += 1; if context_of(g, p) != ctx[t] { bad_r += 1; } }
        let s0 = surprises(&d, e);
        for j0 in 1..HORIZON { let mut e2 = e.clone(); for t in j0..HORIZON { for v in e2.ys[t].iter_mut() { *v += 0.37; } }
            let ctx2 = reid_prefixes(&d, &e2.x0, &e2.ys, K, [0.0; 2]); let s2 = surprises(&d, &e2);
            if (0..=j0.min(K)).any(|j| ctx2[j] != ctx[j]) || (0..j0).any(|t| s2[t].to_bits() != s0[t].to_bits()) { bad_c += 1; } } }
    res.insert("M1c_causal".into(), json!({"reid_checks": n_r, "reid_mismatch": bad_r, "future_leaks": bad_c, "pass": bad_r == 0 && bad_c == 0}));
    // (d) NaN con m = m_max fijo (por β) y M3 en la config máxima, 4 semillas, flujo corto
    let mut cfgs: Vec<(String, Cfg)> = vec![];
    for b in [0.0, 0.5, 0.9] { let mut c = Cfg::v4(); set_m1(&mut c, 0.0, b, 0.0, 0.0, 1.0); c.m1_const = c.m1_mmax; cfgs.push((format!("M1d_mmax_b{b}"), c)); }
    let mut c = Cfg::v4(); set_m3(&mut c, 64, 0.5, 0.05, 128); cfgs.push(("M3d_max".into(), c));
    let mut c = Cfg::v4(); set_m3(&mut c, 32, 0.5, 0.05, 32); cfgs.push(("M3e_default".into(), c));
    let jobs: Vec<(String, Cfg, u64)> = cfgs.iter().flat_map(|(n, c)| seeds.iter().map(move |&s| (n.clone(), c.clone(), s))).chain(seeds.iter().map(|&s| ("v4".to_string(), Cfg::v4(), s))).collect();
    let nd: usize = arg_of(a, "--nd").map(|v| v.parse().unwrap()).unwrap_or(128);
    let out: Vec<serde_json::Value> = jobs.par_iter().map(|(n, c, s)| { let (_, _, nan, w) = trace_cfg(c, *s, &d, nd); json!({"case": n, "seed": format!("0x{s:X}"), "nan": nan, "wall_s": w}) }).collect();
    res.insert("nan_runs".into(), json!({"n_stream": nd, "runs": out, "pass": out.iter().all(|v| v["nan"] == false)}));
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(); assert_eq!(d.raw, raw0);
    res.insert("test_queries".into(), json!(q)); res.insert("orbita_unchanged".into(), json!(true));
    println!("{}", json!({"seeds": seeds.iter().map(|s| format!("0x{s:X}")).collect::<Vec<_>>(), "git_sha": git_sha(), "res": res}));
}
