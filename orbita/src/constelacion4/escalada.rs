//! CONSTELACIÓN v4-ESCALADA (docs/constelacion_v4_escalada/PREREG.md).
//! Variantes separables sobre `Cfg::v4` congelada: P (más capas), A (más ancho), T (más entrenamiento), PAT (todo).
//! Con los tres interruptores apagados la configuración y los flujos son EXACTAMENTE los de v4 (verificado en `cst4esc-paso0`).
//!   orbita cst4esc-paso0 --seed 0xC4E0 --weights W
//!   orbita cst4esc-run --phase dev|confirm|smoke --seed S --weights W --out DIR [--arms v4,P,A,T,PAT] (smoke: --na/--nb opcionales)
use super::*;

/// P: profundidad 3 → 5 capas, sin ensanchar (se repiten los anchos de v4 [16,8,4])
pub const LAYERS_P: [usize; 5] = [16, 8, 8, 4, 4];
/// A: ancho ×2 por capa (neuronas ÓRBITA por capa), misma profundidad
pub const LAYERS_A: [usize; 3] = [32, 16, 8];
/// PA (dentro de PAT): profundidad de P con ancho de A
pub const LAYERS_PA: [usize; 5] = [32, 16, 16, 8, 8];
/// T: flujos de entrenamiento ×T_MULT sobre v4 (N_A = 3072, N_B = 192+192)
pub const T_MULT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Esc { pub p: bool, pub a: bool, pub t: bool }
impl Esc {
    pub const V4: Esc = Esc { p: false, a: false, t: false };
    pub fn arms() -> [(&'static str, Esc); 5] { [("v4", Esc::V4), ("P", Esc { p: true, a: false, t: false }), ("A", Esc { p: false, a: true, t: false }),
        ("T", Esc { p: false, a: false, t: true }), ("PAT", Esc { p: true, a: true, t: true })] }
    pub fn cfg(self) -> Cfg { let mut c = Cfg::v4();
        c.layers = match (self.p, self.a) { (false, false) => c.layers, (true, false) => LAYERS_P.to_vec(), (false, true) => LAYERS_A.to_vec(), (true, true) => LAYERS_PA.to_vec() }; c }
    pub fn streams(self) -> (usize, usize) { if self.t { (N_A * T_MULT, N_NEW * T_MULT) } else { (N_A, N_NEW) } }
}

/// AUROC (Mann–Whitney, empates ½) de la sorpresa = h1_log por episodio en el flujo B; positivo = familia nueva
pub fn auroc(score: &[f64], pos: &[bool]) -> f64 {
    let p: Vec<f64> = score.iter().zip(pos).filter(|x| *x.1).map(|x| *x.0).collect(); let n: Vec<f64> = score.iter().zip(pos).filter(|x| !*x.1).map(|x| *x.0).collect();
    if p.is_empty() || n.is_empty() { return f64::NAN; }
    let mut s = 0.0; for a in &p { for b in &n { s += if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 }; } } s / (p.len() * n.len()) as f64
}
fn n_params(net: &Net) -> usize { net.n_w() + net.wp.len() }

pub const DEV: std::ops::RangeInclusive<u64> = 0xA450..=0xA45F;
pub const CONFIRM: std::ops::RangeInclusive<u64> = 0xB450..=0xB45F;

pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); let phase = arg_of(a, "--phase").unwrap();
    match phase.as_str() { "dev" => assert!(DEV.contains(&seed)), "confirm" => assert!(CONFIRM.contains(&seed)), _ => assert!(phase == "smoke" && (0xC4E0..=0xC4EF).contains(&seed)) }
    if phase != "smoke" { assert!(!a.iter().any(|x| x.starts_with("--") && !["--phase", "--seed", "--weights", "--out"].contains(&x.as_str())), "DEV/CONFIRM: valores congelados"); }
    let wpath = arg_of(a, "--weights").unwrap(); let d = Orbita::from_json(&std::fs::read_to_string(&wpath).unwrap()); let raw0 = d.raw.clone();
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done")); if done.exists() { eprintln!("skip"); return; }
    let only: Option<Vec<String>> = arg_of(a, "--arms").map(|s| s.split(',').map(String::from).collect());
    let t0 = std::time::Instant::now(); let before = cdt_rqm_epr::v4_provenance::snapshot();
    let held = gen_heldout(seed, HPF); let held_known: Vec<Ep> = held.iter().filter(|e| e.fam < N_FAM).cloned().collect();
    let (ex, sha) = e33_examples(seed);
    let mut arms = serde_json::Map::new();
    for (name, esc) in Esc::arms() {
        if let Some(o) = &only { if !o.iter().any(|x| x == name) { continue; } }
        let ta = std::time::Instant::now(); let c = esc.cfg();
        let (mut na, mut nb) = esc.streams();
        if phase == "smoke" { if let Some(v) = arg_of(a, "--na") { na = v.parse().unwrap(); } if let Some(v) = arg_of(a, "--nb") { nb = v.parse().unwrap(); } }
        // los flujos de T son extensiones del mismo generador y semilla que v4 (mismos datos, más episodios)
        let sa = gen_known_stream(seed, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()); let sb = gen_stream(seed, nb, nb);
        let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c);
        let (_, h16a, _) = run_stream(&mut net, &mut cdt, &sa, seed);
        let pre = eval_held(&mut net, &held_known);
        let (h1b, _h16b, fb) = run_stream(&mut net, &mut cdt, &sb, seed);
        let post = eval_held(&mut net, &held); let col = collapse(&mut net, &held); let (e33, e33_v3head) = e33_net2(&mut net, &d, &ex);
        let isnew: Vec<bool> = fb.iter().map(|&f| f >= N_FAM).collect();
        arms.insert(name.into(), json!({"esc": format!("{esc:?}"), "layers": c.layers, "na": na, "nb": nb, "n_params": n_params(&net), "pre_known": pre, "held": post, "collapse": col, "e33": e33, "e33_v3head": e33_v3head,
            "surprise_auroc_h1": auroc(&h1b, &isnew), "streamA_last_q_h16": mean(&h16a[h16a.len() * 3 / 4..]), "nan": net.nan, "hash": format!("{:016x}", net.hash()),
            "eta_scale": net.eta_scale, "sleeps": cdt.log.len(), "prune": [net.n_pruned, net.n_grown, net.n_rescale_row, net.n_rescale_spec], "wall_s": ta.elapsed().as_secs_f64()}));
        eprintln!("0x{seed:X} {name} {:.1}s", ta.elapsed().as_secs_f64());
    }
    let orb = orbita_held(&d, &held); let e33o = e33_orbita(&d, &ex);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    assert_eq!(d.raw, raw0); assert_eq!(q, 0, "consultas TEST ≠ 0");
    let r = json!({"seed": format!("0x{seed:X}"), "phase": phase, "weights": wpath, "orbita": orb, "e33_orbita": e33o, "arms": arms, "test_queries": q, "test_sha256": sha, "git_sha": git_sha(), "wall_s": t0.elapsed().as_secs_f64()});
    std::fs::write(out.join(format!("metrics_0x{seed:X}.json")), serde_json::to_string(&r).unwrap()).unwrap(); std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} ok {:.0}s", t0.elapsed().as_secs_f64());
}

/// Paso 0 de la escalada: (r1) Esc::V4 ≡ Cfg::v4 (cfg idéntica); (r2) brazo v4 de este runner ≡ brazo cst de `cst4-run` bit a bit (hash de pesos y métricas)
/// en un flujo corto; (r3) cada variante sólo cambia lo declarado; (r4) las variantes no producen NaN en el flujo corto.
pub fn paso0(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC4E0".into())); assert!((0xC4E0..=0xC4EF).contains(&seed));
    let n: usize = arg_of(a, "--n").map(|s| s.parse().unwrap()).unwrap_or(48);
    let mut res = serde_json::Map::new();
    let v4 = Cfg::v4(); let e0 = Esc::V4.cfg();
    res.insert("r1_cfg_eq_v4".into(), json!({"pass": format!("{e0:?}") == format!("{v4:?}") && Esc::V4.streams() == (N_A, N_NEW)}));
    let sa = gen_known_stream(seed, XOR_KNOWN, n, &(0..N_FAM).collect::<Vec<_>>()); let sb = gen_stream(seed, n / 6, n / 6);
    let go = |c: &Cfg| { let mut net = Net::new(c, seed); let mut cdt = Cdt::new(seed, c); let (a1, a16, _) = run_stream(&mut net, &mut cdt, &sa, seed); let (b1, b16, _) = run_stream(&mut net, &mut cdt, &sb, seed);
        let mut hm: u64 = 0xcbf29ce484222325; for v in a1.iter().chain(&a16).chain(&b1).chain(&b16) { hm ^= v.to_bits(); hm = hm.wrapping_mul(0x100000001b3); } (net.hash(), hm, net.nan, n_params(&net)) };
    let (wa, ma, _, _) = go(&v4); let (wb, mb, _, _) = go(&e0);
    res.insert("r2_bitwise_v4".into(), json!({"pass": wa == wb && ma == mb, "w_hash": format!("{wa:016x}"), "m_hash": format!("{ma:016x}")}));
    let mut r3 = true; let mut info = serde_json::Map::new();
    for (name, esc) in Esc::arms() { let c = esc.cfg(); let mut c2 = c.clone(); c2.layers = v4.layers.clone();
        r3 &= format!("{c2:?}") == format!("{v4:?}"); r3 &= (c.layers != v4.layers) == (esc.p || esc.a); r3 &= (esc.streams() != (N_A, N_NEW)) == esc.t;
        let t = std::time::Instant::now(); let (_, _, nan, np) = go(&c);
        info.insert(name.into(), json!({"layers": c.layers, "streams": esc.streams(), "n_params": np, "nan_short": nan, "short_s": t.elapsed().as_secs_f64()})); r3 &= !nan; }
    res.insert("r3_only_declared_changes_no_nan".into(), json!({"pass": r3, "arms": info}));
    let ok = res.values().all(|v| v["pass"] == true);
    println!("{}", json!({"seed": format!("0x{seed:X}"), "git_sha": git_sha(), "paso0_ok": ok, "res": res}));
}
