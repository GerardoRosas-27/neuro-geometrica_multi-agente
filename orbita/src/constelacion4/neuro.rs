//! CONSTELACIÓN v4-NEURO (docs/constelacion_v4_neuro/PREREG.md). Mecanismos neuro-inspirados M1..M6 sobre v4; con todos apagados ≡ v4 bit a bit.
//!  M1 disparo por sorpresa (umbral homeostático; si no dispara, la neurona queda en su predicción cacheada F(prior) y no actualiza μ)
//!  M2 periodo refractario (K_REF cuadros tras disparar)
//!  M3 plasticidad de tres factores (A disparó ∧ B disparó después ∧ el error de B bajó) — sólo compuerta ∈[0,1] sobre la regla PC de v4
//!  M4 debilitamiento/poda de conexiones que no reducen el error de B en una ventana; rebrote por sorpresa persistente
//!  M5 consolidación: conexiones longevas útiles pasan a memoria lenta (tasa ×M5_SLOW, exentas de M4)
//!  M6 neuromodulación global: sorpresa de red ↑ tasa de aprendizaje y ↓ umbral de disparo
//!   orbita cst4n-paso0 --seed 0xC4E0 [--na N --nb N]
//!   orbita cst4n-run --stage p1|p2 --phase dev|confirm|smoke --seed S --weights W --out DIR [--arms ..] [--mech M1,M3,..(sólo p2)]
use super::*;

pub const R_TARGET: f64 = 0.5; pub const ETA_H: f64 = 0.05; pub const K_REF: u8 = 1;
pub const M3_EMA: f64 = 0.1; pub const LONG_EMA: f64 = 0.001; pub const WIN: usize = 500;
pub const M4_WEAK: f64 = 0.5; pub const M4_PRUNE_AFTER: u8 = 2; pub const M4_REGROW_RATIO: f64 = 2.0;
pub const M5_SLOW: f64 = 0.1; pub const M5_AGE: u16 = 4;
pub const M6_EMA: f64 = 0.01; pub const M6_MIN: f64 = 0.5; pub const M6_MAX: f64 = 2.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Nm { pub m1: bool, pub m2: bool, pub m3: bool, pub m4: bool, pub m5: bool, pub m6: bool }
impl Nm {
    pub fn from_list(s: &str) -> Nm { let mut n = Nm::default(); for t in s.split(',').filter(|t| !t.is_empty()) { match t { "M1" => n.m1 = true, "M2" => n.m2 = true, "M3" => n.m3 = true, "M4" => n.m4 = true, "M5" => n.m5 = true, "M6" => n.m6 = true, _ => panic!("mecanismo {t}") } } n }
    pub fn all() -> Nm { Nm { m1: true, m2: true, m3: true, m4: true, m5: true, m6: true } }
}
#[derive(Clone, Default, Debug)]
pub struct NState {
    pub theta: Vec<f64>, pub gain: f64, pub gema: f64, pub n_fire: f64, pub n_tot: f64,
    pub e2ema: Vec<Vec<f64>>, pub e2long: Vec<Vec<f64>>, pub credit: Vec<Vec<f64>>, pub weak: Vec<Vec<u8>>, pub age: Vec<Vec<u16>>, pub cons: Vec<Vec<bool>>,
    pub n_weak: usize, pub n_pruned: usize, pub n_grown: usize, pub n_cons: usize, pub n_win: usize,
}
fn nneu(net: &Net) -> usize { net.c.layers.iter().sum() }
fn off(net: &Net, layer1: usize) -> usize { net.c.layers[..layer1 - 1].iter().sum() }

/// M1/M2: decisión de disparo de las neuronas de la capa i (1-based) en el primer paso de inferencia del cuadro (μ = prior)
pub(crate) fn decide(net: &Net, st: &mut PState, i: usize, back: &[f64], eps_up: Option<&Vec<f64>>, fmask: &mut Option<Vec<bool>>, _l: usize) {
    let c = &net.c; let n = nneu(net);
    if i == 1 { st.firedp = st.fired.clone(); *fmask = Some(vec![true; n]); }
    let o = off(net, i); let m = c.m; let mut ns = net.ns.borrow_mut();
    if ns.theta.len() != n { ns.theta = vec![0.0; n]; }
    let gain = if c.nm.m6 && ns.gain > 0.0 { ns.gain } else { 1.0 };
    for k in 0..c.layers[i - 1] {
        let mut s = 0.0; for j in k * m..(k + 1) * m { let g = -back[j] + eps_up.map(|e| e[j]).unwrap_or(0.0); s += g * g; } s /= m as f64;
        let idx = o + k; if ns.theta[idx] == 0.0 { ns.theta[idx] = s.max(1e-300); }
        let mut fire = !c.nm.m1 || !(s <= ns.theta[idx] / gain);
        if c.nm.m2 { if st.cool[idx] > 0 { fire = false; st.cool[idx] -= 1; } else if fire { st.cool[idx] = K_REF; } }
        st.fired[idx] = fire; fmask.as_mut().unwrap()[idx] = fire; ns.n_fire += fire as u8 as f64; ns.n_tot += 1.0;
    }
}
pub(crate) fn acc_fire(a: &mut Acc, st: &PState) {
    if a.afire.is_empty() { a.afire = vec![0.0; st.fired.len()]; a.aelig = vec![0.0; st.fired.len()]; }
    for i in 0..st.fired.len() { a.afire[i] += st.fired[i] as u8 as f64; a.aelig[i] += (st.fired[i] || st.firedp[i]) as u8 as f64; }
}
/// homeostasis del umbral (sólo en entrenamiento): log θ += η_h (disparó − r*)
pub(crate) fn homeo(net: &Net, st: &PState, train: bool) {
    if !(net.c.nm.m1 && train) { return; } let mut ns = net.ns.borrow_mut();
    for i in 0..st.fired.len() { let f = st.fired[i] as u8 as f64; ns.theta[i] *= (ETA_H * (f - R_TARGET)).exp(); }
}
/// M6: ganancia global g = clip(√(E0/EMA(E0)), 0.5, 2) — multiplica la tasa y divide el umbral
pub(crate) fn gain(net: &mut Net, e0: f64, train: bool, imagine: bool, lr: f64) -> f64 {
    let ns = net.ns.get_mut(); if !train || imagine || !e0.is_finite() { return lr; }
    if ns.gema <= 0.0 { ns.gema = e0.max(1e-300); }
    let g = (e0 / ns.gema).sqrt().clamp(M6_MIN, M6_MAX); ns.gema = (1.0 - M6_EMA) * ns.gema + M6_EMA * e0; ns.gain = g; lr * g
}
fn ensure(net: &mut Net) { if net.ns.get_mut().e2ema.len() == net.nl() { return; }
    let rows: Vec<usize> = (0..net.nl()).map(|li| net.dims[li]).collect(); let ws: Vec<usize> = net.w.iter().map(|w| w.len()).collect(); let ns = net.ns.get_mut();
    ns.e2ema = rows.iter().map(|&r| vec![f64::NAN; r]).collect(); ns.e2long = ns.e2ema.clone();
    ns.credit = ws.iter().map(|&n| vec![0.0; n]).collect(); ns.weak = ws.iter().map(|&n| vec![0; n]).collect(); ns.age = ws.iter().map(|&n| vec![0; n]).collect(); ns.cons = ws.iter().map(|&n| vec![false; n]).collect(); }
/// tercer factor: ¿bajó el error de la unidad B (fila) respecto de su EMA?
pub(crate) fn row_dec(net: &mut Net, a: &Acc, np: f64) -> Vec<Vec<bool>> { ensure(net); let ns = net.ns.get_mut();
    ns.e2ema.iter().enumerate().map(|(li, v)| v.iter().enumerate().map(|(r, &m)| !(a.e2[li][r] / np >= m)).collect()).collect() }
pub(crate) fn m3_gate(c: &Cfg, a: &Acc, dec: &[Vec<bool>], li: usize, k: usize, cc: usize, _rows: usize) -> f64 {
    let (r, j) = (k / cc, k % cc); if !dec[li][r] { return 0.0; }
    if a.afire.is_empty() || a.nt <= 0.0 { return 1.0; }
    let o_col: usize = c.layers[..li].iter().sum(); let elig = a.aelig[o_col + j / c.m] / a.nt;
    let fb = if li == 0 { 1.0 } else { let o_row: usize = c.layers[..li - 1].iter().sum(); a.afire[o_row + r / c.m] / a.nt }; elig * fb
}
/// EMAs por fila y ventanas M4/M5 (cada WIN pasos de plasticidad)
pub(crate) fn window(net: &mut Net, a: &Acc, np: f64) {
    let c = net.c.clone(); let steps = net.steps;
    { let ns = net.ns.get_mut(); for li in 0..ns.e2ema.len() { for r in 0..ns.e2ema[li].len() { let e = a.e2[li][r] / np; let m = &mut ns.e2ema[li][r]; *m = if m.is_nan() { e } else { (1.0 - M3_EMA) * *m + M3_EMA * e };
        let lg = &mut ns.e2long[li][r]; *lg = if lg.is_nan() { e } else { (1.0 - LONG_EMA) * *lg + LONG_EMA * e }; } } }
    if !(c.nm.m4 || c.nm.m5) || steps % WIN != 0 { return; }
    let Net { w, cm, act, tau, ns, dims, .. } = net; let ns = ns.get_mut(); ns.n_win += 1;
    for li in 0..w.len() { let cc = dims[li + 1];
        for k in 0..w[li].len() { if cm[li][k] == 0.0 || ns.cons[li][k] { ns.credit[li][k] = 0.0; continue; }
            let pos = ns.credit[li][k] > 0.0;
            if c.nm.m4 && !pos { w[li][k] *= M4_WEAK; ns.weak[li][k] += 1; ns.n_weak += 1; if ns.weak[li][k] >= M4_PRUNE_AFTER { cm[li][k] = 0.0; ns.weak[li][k] = 0; ns.n_pruned += 1; } } else if pos { ns.weak[li][k] = 0; }
            if c.nm.m5 { if pos { ns.age[li][k] += 1; if ns.age[li][k] >= M5_AGE { ns.cons[li][k] = true; ns.n_cons += 1; } } else { ns.age[li][k] = 0; } }
            ns.credit[li][k] = 0.0; }
        if c.nm.m4 { for r in 0..dims[li] { if ns.e2ema[li][r] > M4_REGROW_RATIO * ns.e2long[li][r] {
            if let Some(j) = (0..cc).filter(|&j| cm[li][r * cc + j] == 0.0).max_by(|&x, &y| act[li][r * cc + x].total_cmp(&act[li][r * cc + y])) {
                let k = r * cc + j; cm[li][k] = 1.0; w[li][k] = 0.0; tau[li][k] = c.tau_min; ns.weak[li][k] = 0; ns.age[li][k] = 0; ns.credit[li][k] = 0.0; ns.n_grown += 1; } } } }
    }
}

// ─── protocolo ───────────────────────────────────────────────────────────────
pub fn auroc(score: &[f64], pos: &[bool]) -> f64 {
    let p: Vec<f64> = score.iter().zip(pos).filter(|x| *x.1).map(|x| *x.0).collect(); let n: Vec<f64> = score.iter().zip(pos).filter(|x| !*x.1).map(|x| *x.0).collect();
    if p.is_empty() || n.is_empty() { return f64::NAN; }
    let mut s = 0.0; for a in &p { for b in &n { s += if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 }; } } s / (p.len() * n.len()) as f64
}
fn n_params(net: &Net) -> usize { net.n_w() + net.wp.len() }
pub fn arms_p1() -> Vec<(&'static str, Nm)> { vec![("v4", Nm::default()), ("M1", Nm::from_list("M1")), ("M2", Nm::from_list("M2")), ("M3", Nm::from_list("M3")),
    ("M4", Nm::from_list("M4")), ("M5", Nm::from_list("M5")), ("M6", Nm::from_list("M6")), ("ALL", Nm::all())] }
pub fn seeds(stage: &str, phase: &str) -> std::ops::RangeInclusive<u64> { match (stage, phase) { ("p1", "dev") => 0xA460..=0xA46F, ("p1", "confirm") => 0xB460..=0xB46F,
    ("p2", "dev") => 0xA470..=0xA47F, ("p2", "confirm") => 0xB470..=0xB47F, (_, "smoke") => 0xC4E0..=0xC4EF, _ => panic!("fase") } }
fn fire_frac(net: &mut Net) -> f64 { let ns = net.ns.get_mut(); let f = if ns.n_tot > 0.0 { ns.n_fire / ns.n_tot } else { 1.0 }; ns.n_fire = 0.0; ns.n_tot = 0.0; f }

/// p1: entrena v4 en el flujo A UNA vez por semilla (pesos v4 entrenados, reutilizados); cada brazo clona esa red y activa sus mecanismos desde el flujo B.
/// p2: cada brazo entrena desde cero (flujo A + B) con sus mecanismos (NEU = --mech) contra v4.
pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); let phase = arg_of(a, "--phase").unwrap(); let stage = arg_of(a, "--stage").unwrap();
    assert!(seeds(&stage, &phase).contains(&seed), "semilla fuera de rango preregistrado");
    if phase != "smoke" { assert!(!a.iter().any(|x| x.starts_with("--") && !["--phase", "--seed", "--weights", "--out", "--arms", "--stage", "--mech"].contains(&x.as_str())), "DEV/CONFIRM: valores congelados"); }
    let wpath = arg_of(a, "--weights").unwrap(); let d = Orbita::from_json(&std::fs::read_to_string(&wpath).unwrap()); let raw0 = d.raw.clone();
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done")); if done.exists() { eprintln!("skip"); return; }
    let only: Option<Vec<String>> = arg_of(a, "--arms").map(|s| s.split(',').map(String::from).collect());
    let t0 = std::time::Instant::now(); let before = cdt_rqm_epr::v4_provenance::snapshot();
    let held = gen_heldout(seed, HPF); let held_known: Vec<Ep> = held.iter().filter(|e| e.fam < N_FAM).cloned().collect();
    let (ex, sha) = e33_examples(seed);
    let (mut na, mut nb) = (N_A, N_NEW);
    if phase == "smoke" { if let Some(v) = arg_of(a, "--na") { na = v.parse().unwrap(); } if let Some(v) = arg_of(a, "--nb") { nb = v.parse().unwrap(); } }
    let sa = gen_known_stream(seed, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()); let sb = gen_stream(seed, nb, nb);
    let arms: Vec<(String, Nm)> = if stage == "p1" { arms_p1().into_iter().map(|(n, m)| (n.to_string(), m)).collect() }
        else { let mech = arg_of(a, "--mech").expect("p2 requiere --mech"); vec![("v4".into(), Nm::default()), ("NEU".into(), Nm::from_list(&mech))] };
    let c4 = Cfg::v4();
    // p1: entrenamiento v4 compartido (flujo A) — reutilizado por todos los brazos
    let shared = if stage == "p1" { let ta = std::time::Instant::now(); let mut net = Net::new(&c4, seed); let mut cdt = Cdt::new(seed, &c4);
        let (_, h16a, _) = run_stream(&mut net, &mut cdt, &sa, seed); let pre = eval_held(&mut net, &held_known);
        Some((net, cdt, h16a, pre, ta.elapsed().as_secs_f64())) } else { None };
    let mut res = serde_json::Map::new();
    for (name, nm) in arms {
        if let Some(o) = &only { if !o.iter().any(|x| *x == name) { continue; } }
        let ta = std::time::Instant::now();
        let (mut net, mut cdt, h16a, pre, wa) = match &shared { Some((n, c, h, p, w)) => { let mut n = n.clone(); n.c.nm = nm; n.ns = Default::default(); (n, c.clone(), h.clone(), p.clone(), *w) }
            None => { let mut c = c4.clone(); c.nm = nm; let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c);
                let (_, h16a, _) = run_stream(&mut net, &mut cdt, &sa, seed); let fa = fire_frac(&mut net); let pre = eval_held(&mut net, &held_known); let _ = fa; (net, cdt, h16a, pre, 0.0) } };
        let _ = fire_frac(&mut net);
        let (h1b, _h16b, fb) = run_stream(&mut net, &mut cdt, &sb, seed); let ff_b = fire_frac(&mut net);
        let post = eval_held(&mut net, &held); let ff_eval = fire_frac(&mut net);
        let col = collapse(&mut net, &held); let (e33, e33_v3head) = e33_net2(&mut net, &d, &ex);
        let isnew: Vec<bool> = fb.iter().map(|&f| f >= N_FAM).collect();
        let ns = net.ns.borrow().clone(); let mut th = ns.theta.clone(); th.sort_by(|x, y| x.total_cmp(y));
        let cons_frac = { let (mut a1, mut b1) = (0.0, 0.0); for li in 0..ns.cons.len() { for k in 0..ns.cons[li].len() { if net.cm[li][k] != 0.0 { b1 += 1.0; a1 += ns.cons[li][k] as u8 as f64; } } } if b1 > 0.0 { a1 / b1 } else { 0.0 } };
        res.insert(name.clone(), json!({"nm": format!("{nm:?}"), "layers": net.c.layers, "na": na, "nb": nb, "n_params": n_params(&net), "n_active_conn": net.cm.iter().flatten().filter(|&&v| v != 0.0).count(),
            "pre_known": pre, "held": post, "collapse": col, "e33": e33, "e33_v3head": e33_v3head, "surprise_auroc_h1": auroc(&h1b, &isnew),
            "fire_frac_streamB": ff_b, "fire_frac_eval": ff_eval, "theta_med": th.get(th.len() / 2).copied().unwrap_or(f64::NAN),
            "m4": {"weakened": ns.n_weak, "pruned": ns.n_pruned, "grown": ns.n_grown}, "m5": {"consolidated": ns.n_cons, "frac_active_consolidated": cons_frac}, "windows": ns.n_win,
            "streamA_last_q_h16": mean(&h16a[h16a.len() * 3 / 4..]), "nan": net.nan, "hash": format!("{:016x}", net.hash()),
            "eta_scale": net.eta_scale, "sleeps": cdt.log.len(), "prune": [net.n_pruned, net.n_grown, net.n_rescale_row, net.n_rescale_spec], "wall_s": ta.elapsed().as_secs_f64(), "shared_streamA_s": wa}));
        eprintln!("0x{seed:X} {name} {:.1}s", ta.elapsed().as_secs_f64());
    }
    let orb = orbita_held(&d, &held); let e33o = e33_orbita(&d, &ex);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    assert_eq!(d.raw, raw0); assert_eq!(q, 0, "consultas TEST ≠ 0");
    let r = json!({"seed": format!("0x{seed:X}"), "stage": stage, "phase": phase, "weights": wpath, "orbita": orb, "e33_orbita": e33o, "arms": res, "test_queries": q, "test_sha256": sha, "git_sha": git_sha(), "wall_s": t0.elapsed().as_secs_f64()});
    std::fs::write(out.join(format!("metrics_0x{seed:X}.json")), serde_json::to_string(&r).unwrap()).unwrap(); std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} ok {:.0}s", t0.elapsed().as_secs_f64());
}

/// Paso 0: (a) Cfg::v4 con nm=off es la v4 congelada (todos los flags falsos); (b) brazo v4 (p1, red clonada) ≡ v4 en línea recta (Net::new→A→B) bit a bit en hash de pesos y curvas;
/// (c) imprime hash/métricas del brazo v4 para compararlo contra el binario de exp/constelacion-v4-escalada (`cst4esc-run --phase smoke`) en el script; (d) cada mecanismo sin NaN en flujo corto.
pub fn paso0(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC4E0".into())); assert!((0xC4E0..=0xC4EF).contains(&seed));
    let na: usize = arg_of(a, "--na").map(|s| s.parse().unwrap()).unwrap_or(48); let nb: usize = arg_of(a, "--nb").map(|s| s.parse().unwrap()).unwrap_or(8);
    let mut res = serde_json::Map::new(); let c4 = Cfg::v4();
    res.insert("a_cfg_off".into(), json!({"pass": c4.nm == Nm::default()}));
    let sa = gen_known_stream(seed, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()); let sb = gen_stream(seed, nb, nb);
    let hv = |v: &[f64], h: &mut u64| { for x in v { *h ^= x.to_bits(); *h = h.wrapping_mul(0x100000001b3); } };
    // recta
    let mut n1 = Net::new(&c4, seed); let mut k1 = Cdt::new(seed, &c4); let (a1, a16, _) = run_stream(&mut n1, &mut k1, &sa, seed); let (b1, b16, _) = run_stream(&mut n1, &mut k1, &sb, seed);
    let mut m1: u64 = 0xcbf29ce484222325; for v in [&a1, &a16, &b1, &b16] { hv(v, &mut m1); }
    // clonada con nm off explícito y NState reiniciado
    let mut n2 = Net::new(&c4, seed); let mut k2 = Cdt::new(seed, &c4); let (a2, a26, _) = run_stream(&mut n2, &mut k2, &sa, seed);
    let mut n3 = n2.clone(); n3.c.nm = Nm::default(); n3.ns = Default::default(); let mut k3 = k2.clone(); let (b2, b26, _) = run_stream(&mut n3, &mut k3, &sb, seed);
    let mut m2: u64 = 0xcbf29ce484222325; for v in [&a2, &a26, &b2, &b26] { hv(v, &mut m2); }
    res.insert("b_clone_eq_straight".into(), json!({"pass": n1.hash() == n3.hash() && m1 == m2, "w_hash": format!("{:016x}", n1.hash()), "m_hash": format!("{m1:016x}")}));
    let held = gen_heldout(seed, HPF); let ev = eval_held(&mut n1, &held);
    res.insert("c_v4_eval".into(), json!({"held": ev, "w_hash_after_eval": format!("{:016x}", n1.hash())}));
    let mut ok_d = true; let mut info = serde_json::Map::new();
    for (name, nm) in arms_p1() { let mut n = n2.clone(); n.c.nm = nm; n.ns = Default::default(); let mut k = k2.clone(); let _ = fire_frac(&mut n);
        let t = std::time::Instant::now(); let (bb, _, _) = run_stream(&mut n, &mut k, &sb, seed); let ff = fire_frac(&mut n); let e = eval_held(&mut n, &held);
        ok_d &= !n.nan; info.insert(name.into(), json!({"nan": n.nan, "fire_frac": ff, "h1_mean_B": mean(&bb), "held": e, "hash": format!("{:016x}", n.hash()), "s": t.elapsed().as_secs_f64()})); }
    res.insert("d_no_nan".into(), json!({"pass": ok_d, "arms": info}));
    let ok = res.values().all(|v| v.get("pass").map(|p| p == true).unwrap_or(true));
    println!("{}", json!({"seed": format!("0x{seed:X}"), "git_sha": git_sha(), "paso0_ok": ok, "res": res}));
}
