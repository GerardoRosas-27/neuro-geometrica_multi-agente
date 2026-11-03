//! CST vs ÓRBITA con entrenamiento igualado (docs/cst_vs_orbita/PREREG.md + ENMIENDA_1.md).
//!   orbita cvo-export --seed S --out DIR        transiciones de los flujos A+B de CST (mismo orden) para entrenar ÓRBITA-eq (sin replay)
//!   orbita cvo-eval --phase dev|confirm|tune --seed S --weights W_eq [--orig W_v1] --out DIR
//! Evaluación homologada: ambos modelos reciben exactamente la misma información (k=4 cuadros observados, sin etiqueta de familia).
use super::*;
use crate::latent::{ctx8, fit_gn, lat_step, npar};

pub const DELTAS: [usize; 7] = [1, 2, 4, 8, 16, 32, 60]; // E33-k4: pasos tras el último cuadro observado (t = 4+Δ, hasta t = 64)
pub const HZ_CTX: [usize; 7] = [1, 2, 4, 8, 16, 32, 64]; // E33-ctx (protocolo ÓRBITA original)
pub const LAT_LAM0: f64 = 1e-4;
pub const CDIM: usize = 10;

fn put(f: &mut Vec<u8>, v: &[f64]) { for x in v { f.extend_from_slice(&x.to_le_bytes()); } }
/// contexto de entrenamiento de ÓRBITA-eq: conocidas = contexto verdadero (como ÓRBITA-v1); nuevas = vector 0 (sin etiqueta)
fn tr_ctx(e: &Ep) -> Vec<f64> { if e.fam < N_FAM { context_of(e.fam, e.p) } else { vec![0.0; CDIM] } }
fn streams(seed: u64) -> (Vec<Ep>, Vec<Ep>) {
    (gen_known_stream(seed, XOR_KNOWN, N_A, &(0..N_FAM).collect::<Vec<_>>()), gen_stream(seed, N_NEW, N_KNOWN_B))
}

pub fn export(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); let out = PathBuf::from(arg_of(a, "--out").unwrap()); std::fs::create_dir_all(&out).unwrap();
    let (sa, sb) = streams(seed); let mut tb = vec![]; let mut n = 0usize; let (mut n_new, mut n_known) = (0usize, 0usize);
    for e in sa.iter().chain(&sb) { let c = tr_ctx(e); for t in 0..HORIZON { put(&mut tb, e.frame(t)); put(&mut tb, &c); put(&mut tb, e.frame(t + 1)); n += 1; if e.fam < N_FAM { n_known += 1 } else { n_new += 1 } } }
    std::fs::write(out.join("train.bin"), &tb).unwrap();
    let meta = json!({"seed": format!("0x{seed:X}"), "n_train": n, "n_train_known": n_known, "n_train_new": n_new, "episodes": sa.len() + sb.len(), "zdim": 2 * K_POINTS, "cdim": CDIM, "replay_fraction": 0.0, "pool": 0, "ks": [1, 2, 4, 8]});
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    println!("cvo-export 0x{seed:X}: transiciones={n} (conocidas {n_known}, nuevas {n_new})");
}

/// CST: observa los cuadros (W^F se adapta, sin plasticidad lenta) y hace rollout reinyectado n pasos (idéntico a run_episode en eval)
fn cst_roll(net: &mut Net, obs: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let np = obs[0].len() / 2; net.begin_episode(); let mut sts: Vec<PState> = (0..np).map(|_| net.fresh_state()).collect(); let nan0 = net.nan;
    for x in obs { constelacion_step(net, &mut sts, x, false, 0.0, false); }
    let mut s2 = sts.clone(); let mut out = vec![];
    for _ in 0..n { let p: Vec<f64> = s2.iter().flat_map(|s| net.predict1(s)).collect(); out.push(p.clone()); if !fin(&p) { break; } step_ex(net, &mut s2, &p, false, 0.0, false, true); }
    while out.len() < n { out.push(vec![f64::NAN; 2 * np]); }
    net.nan = nan0; out
}
#[derive(Clone, Copy, PartialEq)] pub enum OMode { Frozen, Lat }
/// ÓRBITA: contexto dado (E33-ctx) o re-identificado con los k cuadros observados (p̂ = 0, sin parámetros verdaderos); LAT = LAT_EP ajustado en los mismos k cuadros
fn orb_roll(d: &Orbita, obs: &[Vec<f64>], n: usize, ctx: Option<&[f64]>, mode: OMode) -> Vec<Vec<f64>> {
    let k = obs.len() - 1; let mut z = obs[k].clone(); let mut out = vec![];
    match ctx {
        Some(c) => { for _ in 0..n { z = d.step(&z, c); out.push(z.clone()); } }
        None => {
            let ys: Vec<Vec<f64>> = obs[1..].to_vec(); let (g, pp, _) = reid(d, &obs[0], &ys, k, [0.0, 0.0]);
            if mode == OMode::Lat { let cx = ctx8(d, pp); let th = fit_gn(&|th, z| lat_step(d, &cx.cps[g], &cx.cs[g], th, z), &vec![0.0; npar()], LAT_LAM0, &obs[0], &ys, k);
                for _ in 0..n { z = lat_step(d, &cx.cps[g], &cx.cs[g], &th, &z); out.push(z.clone()); if !fin(&z) { break; } } }
            else { let c = context_of(g, pp); for _ in 0..n { z = d.step(&z, &c); out.push(z.clone()); if !fin(&z) { break; } } }
        }
    }
    while out.len() < n { out.push(vec![f64::NAN; obs[0].len()]); }
    out
}
fn ok(p: &[f64], y: &[f64]) -> bool { fin(p) && mt::correct(p, y) }
fn region(accs: &[f64], hs: &[usize]) -> usize { let mut r = 0; for (a, &h) in accs.iter().zip(hs) { if *a >= 0.5 { r = h } else { break } } r }

/// evaluación homologada de un predictor genérico: f(obs, n, ctx) -> predicciones de los n cuadros siguientes
fn eval_all(f: &mut dyn FnMut(&[Vec<f64>], usize, Option<&[f64]>) -> Vec<Vec<f64>>, held: &[Ep], e33: &[Example], test: &[Example], ctx_ok: bool) -> serde_json::Value {
    let mut r = serde_json::Map::new();
    for (name, known) in [("known", true), ("new", false)] {
        let errs: Vec<f64> = held.iter().filter(|e| (e.fam < N_FAM) == known).map(|e| { let obs: Vec<Vec<f64>> = (0..=K).map(|t| e.frame(t).to_vec()).collect();
            let p = f(&obs, HORIZON - K, None); clamp_err(&p[HORIZON - K - 1], &e.ys[HORIZON - 1]) }).collect();
        r.insert(format!("{name}_acc16"), json!(errs.iter().filter(|&&x| x < mt::REL_ERR_OK).count() as f64 / errs.len() as f64));
        r.insert(format!("{name}_med_log"), json!(median(errs.iter().map(|&x| log_rel(x)).collect())));
    }
    let nmax = *DELTAS.last().unwrap(); let mut acc = vec![0.0; DELTAS.len()];
    for e in e33 { let obs: Vec<Vec<f64>> = (0..=K).map(|t| target_h(e, t)).collect(); let p = f(&obs, nmax, None);
        for (j, &dl) in DELTAS.iter().enumerate() { acc[j] += ok(&p[dl - 1], &target_h(e, K + dl)) as u8 as f64; } }
    let acc: Vec<f64> = acc.iter().map(|v| v / e33.len() as f64).collect(); let rg = region(&acc, &DELTAS);
    r.insert("e33_k4".into(), json!({"acc": acc, "deltas": DELTAS, "region": rg, "pass": rg >= 8, "n": e33.len()}));
    if ctx_ok {
        let mut acc = vec![0.0; HZ_CTX.len()];
        for e in e33 { let p = f(&[e.input.clone()], 64, Some(&e.context)); for (j, &h) in HZ_CTX.iter().enumerate() { acc[j] += ok(&p[h - 1], &target_h(e, h)) as u8 as f64; } }
        let acc: Vec<f64> = acc.iter().map(|v| v / e33.len() as f64).collect(); let rg = region(&acc, &HZ_CTX);
        r.insert("e33_ctx".into(), json!({"acc": acc, "horizons": HZ_CTX, "region": rg, "pass": rg >= 8, "n": e33.len()}));
        let c1 = test.iter().filter(|e| ok(&f(&[e.input.clone()], 1, Some(&e.context))[0], &target_h(e, 1))).count();
        r.insert("onestep_ctx_acc".into(), json!(c1 as f64 / test.len() as f64));
    }
    serde_json::Value::Object(r)
}

pub fn eval(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); let phase = arg_of(a, "--phase").unwrap();
    match phase.as_str() { "dev" => assert!((0xA400..=0xA40F).contains(&seed)), "confirm" => assert!((0xB400..=0xB40F).contains(&seed)),
        _ => assert!(phase == "tune" && ((0xC400..=0xC4FF).contains(&seed) || (0xD400..=0xD4FF).contains(&seed)), "tuning sólo en 0xC4xx/0xD4xx") }
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done")); if done.exists() { eprintln!("skip"); return; }
    let t0 = std::time::Instant::now(); let before = cdt_rqm_epr::v4_provenance::snapshot();
    let weq = arg_of(a, "--weights").unwrap(); let deq = Orbita::from_json(&std::fs::read_to_string(&weq).unwrap()); let raw0 = deq.raw.clone();
    let dorig = arg_of(a, "--orig").map(|p| Orbita::from_json(&std::fs::read_to_string(p).unwrap()));
    let held = gen_heldout(seed, HPF); let (sa, sb) = streams(seed);
    let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS)); s.verify_test().expect("TEST sealed");
    let test: Vec<Example> = s.ds.test.clone(); let e33: Vec<Example> = test.iter().filter(|e| ITERABLE_FAMS.contains(&e.family_id)).cloned().collect();
    let only: Option<Vec<String>> = arg_of(a, "--arms").map(|s| s.split(',').map(String::from).collect());
    let want = |n: &str| only.as_ref().map_or(true, |o| o.iter().any(|x| x == n));
    let mut arms = serde_json::Map::new();
    // CST (v4 sin orb_prior: entrenado desde cero en los flujos A+B; = cst4 v4 salvo la cabeza ÓRBITA) y no_fast
    for (name, fast) in [("cst", true), ("cst_no_fast", false)] {
        if !want(name) { continue; }
        let ta = std::time::Instant::now(); let mut c = Cfg::v4(); c.orb_prior = false; c.fast = fast;
        let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c);
        run_stream(&mut net, &mut cdt, &sa, seed); run_stream(&mut net, &mut cdt, &sb, seed);
        let replay_eps: usize = cdt.log.iter().map(|l| l["pos"].as_u64().unwrap() as usize + l["neg"].as_u64().unwrap() as usize).sum();
        let samples = (sa.len() + sb.len() + replay_eps) * HORIZON;
        let col = collapse(&mut net, &held);
        let mut ev = eval_all(&mut |obs, n, _c| cst_roll(&mut net, obs, n), &held, &e33, &test, true);
        // CST+prior (informativo, sólo E33-ctx): x̂(1) = ÓRBITA-eq(x0, c) se inyecta como cuadro observado y CST continúa
        if name == "cst" {
            let mut acc = vec![0.0; HZ_CTX.len()];
            for e in &e33 { let x1 = deq.step(&e.input, &e.context); let mut p = vec![x1.clone()]; p.extend(cst_roll(&mut net, &[e.input.clone(), x1], 63));
                for (j, &h) in HZ_CTX.iter().enumerate() { acc[j] += ok(&p[h - 1], &target_h(e, h)) as u8 as f64; } }
            let acc: Vec<f64> = acc.iter().map(|v| v / e33.len() as f64).collect(); let rg = region(&acc, &HZ_CTX);
            arms.insert("cst_prior".into(), json!({"e33_ctx": {"acc": acc, "horizons": HZ_CTX, "region": rg, "pass": rg >= 8, "n": e33.len()}}));
        }
        let o = ev.as_object_mut().unwrap(); o.insert("collapse".into(), col); o.insert("nan".into(), json!(net.nan)); o.insert("train_samples".into(), json!(samples));
        o.insert("replay_episodes".into(), json!(replay_eps)); o.insert("sleeps".into(), json!(cdt.log.len())); o.insert("hash".into(), json!(format!("{:016x}", net.hash()))); o.insert("wall_s".into(), json!(ta.elapsed().as_secs_f64()));
        arms.insert(name.into(), ev); eprintln!("0x{seed:X} {name} {:.1}s", ta.elapsed().as_secs_f64());
    }
    let mut orbs: Vec<(&str, &Orbita, OMode)> = vec![("orbita_eq", &deq, OMode::Frozen), ("orbita_eq_lat", &deq, OMode::Lat)];
    if let Some(d0) = &dorig { orbs.push(("orbita_v1", d0, OMode::Frozen)); orbs.push(("orbita_v1_lat", d0, OMode::Lat)); }
    for (name, d, mode) in orbs { if !want(name) { continue; } let ta = std::time::Instant::now();
        let mut ev = eval_all(&mut |obs, n, c| orb_roll(d, obs, n, c, mode), &held, &e33, &test, mode == OMode::Frozen);
        ev.as_object_mut().unwrap().insert("wall_s".into(), json!(ta.elapsed().as_secs_f64())); arms.insert(name.into(), ev); eprintln!("0x{seed:X} {name} {:.1}s", ta.elapsed().as_secs_f64()); }
    if want("persist") { arms.insert("persist".into(), eval_all(&mut |obs, n, _c| vec![obs[obs.len() - 1].clone(); n], &held, &e33, &test, true)); }
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    assert_eq!(deq.raw, raw0); assert_eq!(q, 0, "consultas TEST ≠ 0");
    let wmeta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&weq).unwrap()).unwrap();
    let r = json!({"seed": format!("0x{seed:X}"), "phase": phase, "weights_eq": weq, "orbita_eq_train": wmeta["train"], "orbita_eq_final_l1": wmeta["final_l1"], "arms": arms, "test_queries": q, "test_sha256": s.manifest.test_sha256, "git_sha": git_sha(), "wall_s": t0.elapsed().as_secs_f64()});
    std::fs::write(out.join(format!("metrics_0x{seed:X}.json")), serde_json::to_string(&r).unwrap()).unwrap(); std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} ok {:.0}s", t0.elapsed().as_secs_f64());
}
