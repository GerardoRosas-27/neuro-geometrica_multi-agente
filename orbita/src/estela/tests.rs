//! Paso 0 · paridad entrenamiento/evaluación (el fallo del experimento Bytes: la ruta de evaluación
//! no era la de entrenamiento). Estas pruebas fallan si las dos rutas divergen en un solo bit.
use super::*;

fn orb(seed: &str) -> Orbita {
    let p = format!("{}/../artifacts/orbita/dev/weights_{seed}.json", env!("CARGO_MANIFEST_DIR"));
    Orbita::from_json(&std::fs::read_to_string(p).unwrap())
}

/// (1) Con r_φ = 0, h_ψ = 0, P = [I 0] y el lifting de ÓRBITA, ESTELA reproduce ÓRBITA bit a bit (16 pasos, conocidas y nuevas).
/// (2) El forward con cinta (el que usa la pérdida de entrenamiento) da los mismos bits que el forward f64 de evaluación.
#[test]
fn paso0_bitwise_init_and_tape() {
    for s in ["0xA400", "0xA40F"] {
        let d = orb(s); let l = Lay::new(d.hg);
        let (a, b, n) = paso0_check(&d, &l, 0xA400, 2);
        assert!(n > 10_000, "pocas comparaciones: {n}");
        assert_eq!(a, 0, "ESTELA@init ≠ ÓRBITA en {a} valores ({s})");
        assert_eq!(b, 0, "cinta ≠ f64 en {b} valores ({s})");
    }
}

/// (3) Las predicciones q̂ que entran en la pérdida de entrenamiento (batch_loss) son, bit a bit,
/// las que produce `rollout` (evaluación) con los mismos parámetros, tras entrenar unos pasos (r_φ, h_ψ, g ≠ init).
#[test]
fn paso0_loss_predictions_equal_eval_rollout() {
    let d = orb("0xA400"); let l = Lay::new(d.hg);
    let data = gen_known(0xA400, 0x1234, 64);
    for v in Variant::all() {
        let (th, _) = train(&d, &l, v, 0xA400, &data, 20);
        let eps: Vec<(&Ep, usize)> = vec![(&data[3], 0), (&data[10], 5), (&data[17], 12)];
        let mut rng = Xoshiro256StarStar::seed_from_u64(1); let dirs = rand_dirs(&mut rng);
        ad::reset(); let tv: Vec<Var> = th.iter().map(|&x| ad::leaf(x)).collect();
        let mut preds = vec![]; let _ = batch_loss(&d, &tv, &l, v, &eps, &dirs, Some(&mut preds)); ad::reset();
        let mut j = 0;
        for (e, t0) in &eps {
            let x0 = if *t0 == 0 { e.x0.clone() } else { e.ys[t0 - 1].clone() };
            let fz = Frozen::new(&d, &e.c, v);
            let tr = rollout(&th, &l, &fz, &x0, HTRAIN, None);
            for t in 0..HTRAIN { for (p, q) in preds[j].iter().zip(&tr[t]) { assert_eq!(p.to_bits(), q.to_bits(), "{:?} ep{j} t{t}", v); } j += 1; }
        }
    }
}

/// (4) Parámetros escritos y releídos (JSON) dan predicciones idénticas: la evaluación usa lo que el entrenamiento produjo.
#[test]
fn paso0_params_roundtrip() {
    let d = orb("0xA400"); let l = Lay::new(d.hg);
    let data = gen_known(0xA400, 0x99, 32);
    let (th, _) = train(&d, &l, Variant::Full, 0xA400, &data, 10);
    let th2 = params_from_json(&params_to_json(&th));
    assert!(th.iter().zip(&th2).all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// (5) Mismo formato de datos en entrenamiento, calibración y evaluación (mismo generador, 6 puntos, horizonte 16).
#[test]
fn paso0_same_data_format() {
    let tr = gen_known(0xA400, XOR_TRAIN, 16); let he = gen_heldout(0xA400, 2);
    for e in tr.iter().chain(&he) { assert_eq!(e.x0.len(), 2 * K_POINTS); assert_eq!(e.ys.len(), HORIZON); assert!(e.ys.iter().all(|y| y.len() == 2 * K_POINTS)); assert_eq!(e.c.len(), 10); }
    assert!(tr.iter().all(|e| e.fam < N_FAM));
}

/// (6) El gradiente de la cinta coincide con diferencias finitas (la pérdida que se optimiza es la que se dice).
#[test]
fn autodiff_matches_finite_differences() {
    let d = orb("0xA400"); let l = Lay::new(d.hg);
    let data = gen_known(0xA400, 0x77, 8);
    let mut rng = Xoshiro256StarStar::seed_from_u64(5);
    let th: Vec<f64> = model::init(&d, &l, 1).iter().map(|x| x + 0.02 * (rng.gen::<f64>() - 0.5)).collect();
    let eps: Vec<(&Ep, usize)> = vec![(&data[1], 2), (&data[2], 0)]; let dirs = rand_dirs(&mut rng);
    for v in [Variant::Full, Variant::Rec] {
        let (g, _, _) = grad(&d, &th, &l, v, &eps, &dirs);
        let f = |t: &[f64]| { ad::reset(); let tv: Vec<Var> = t.iter().map(|&x| ad::leaf(x)).collect(); let r = batch_loss_tgt(&d, &tv, &th, &l, v, &eps, &dirs, None).0.v(); ad::reset(); r };
        for &i in &[l.w3 + 5, l.b4 + 1, l.r1 + 3, l.r2 + 7, l.e1 + 2, l.e2 + 4, l.pr + 9] {
            let h = 1e-6; let mut a = th.clone(); a[i] += h; let mut b = th.clone(); b[i] -= h;
            let fd = (f(&a) - f(&b)) / (2.0 * h);
            assert!((fd - g[i]).abs() <= 1e-5 * (1.0 + fd.abs()), "{v:?} i={i} fd={fd} ad={}", g[i]);
        }
    }
}

/// diagnóstico (no es prueba): traza de entrenamiento fuera de DEV/CONFIRM (seed 0xC400, pesos ÓRBITA 0xA400)
#[test]
#[ignore]
fn diag_trace_c400() {
    let d = orb("0xA400"); let l = Lay::new(d.hg);
    let data = gen_known(0xC400, XOR_TRAIN, N_TRAIN);
    let vs: Vec<Variant> = Variant::all().into_iter().filter(|v| std::env::var("DIAG_V").map_or(true, |s| s.split(',').any(|x| x == v.name()))).collect();
    for v in vs { let _ = train(&d, &l, v, 0xC400, &data, STEPS); }
}

/// ENMIENDA_1 §2: ESTELA-r0 «entrenado» = ÓRBITA congelada, bit a bit (conocidas y nuevas, contexto de evaluación)
#[test]
fn r0_equals_frozen_orbita_bitwise() {
    let d = orb("0xA400"); let l = Lay::new(d.hg);
    let data = gen_known(0xC400, XOR_TRAIN, 64);
    let (th, _) = train(&d, &l, Variant::R0, 0xC400, &data, 50);
    assert!(th.iter().zip(&model::init(&d, &l, 0xC400)).all(|(a, b)| a.to_bits() == b.to_bits()));
    let mut n = 0;
    for e in &gen_heldout(0xC400, 2) {
        let (cg, _) = ctx_for_eval(&d, e, 4); let fz = Frozen::new(&d, &cg, Variant::R0);
        let tr = rollout(&th, &l, &fz, &e.x0, 16, None); let mut z = e.x0.clone();
        for t in 0..16 { z = d.step(&z, &cg); for (p, q) in tr[t].iter().zip(&z) { assert_eq!(p.to_bits(), q.to_bits()); n += 1; } }
    }
    assert!(n > 1000);
}

/// ENMIENDA_1 §1.1: el codificador no cambia al entrenar (todas las variantes)
#[test]
fn encoder_frozen_after_training() {
    let d = orb("0xA400"); let l = Lay::new(d.hg);
    let data = gen_known(0xC400, XOR_TRAIN, 64); let th0 = model::init(&d, &l, 0xC400);
    for v in Variant::all() { let (th, _) = train(&d, &l, v, 0xC400, &data, 30);
        assert!((l.w3..l.r1).all(|i| th[i].to_bits() == th0[i].to_bits()), "{v:?}");
        if v != Variant::R0 { assert!((l.r1..l.n).any(|i| th[i] != th0[i]), "{v:?} no entrena"); } }
}
