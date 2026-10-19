//! Otras familias (docs/preregistro_otras_familias.md) y paso de la demo (docs/demo_orbita.md). No lee ds.test.
use crate::model::Orbita;
use cdt_rqm_epr::v4_dataset::{apply_rule, context_of, make_object, ObjectClass, N_FAM};
use cdt_rqm_epr::v4_metrics as m;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
pub const NF: usize = 10; // 0–7 v4, 8 = B1 giro ondulado (rotación + no linealidad suave), 9 = B2 espiral (de mem); 10 pulso radial, 11 cizalla saturada (sólo con --nf 12)
fn rot(t: f64, x: f64, y: f64) -> (f64, f64) { (t.cos() * x - t.sin() * y, t.sin() * x + t.cos() * y) }
pub fn rule(f: usize, p: [f64; 2], o: &[f64]) -> Vec<f64> {
    if f < N_FAM { return apply_rule(f, p, o); }
    let mut v = o.to_vec();
    for i in 0..o.len() / 2 {
        let (x, y) = (o[2 * i], o[2 * i + 1]);
        let (a, b) = if f == 8 { let (u, w) = rot(0.6 * p[0], x, y); (u + 0.25 * p[1] * (2.0 * w).sin(), w + 0.25 * p[1] * (2.0 * u).sin()) }
            else if f == 9 { let r = (x * x + y * y).sqrt(); let (u, w) = rot(0.4 * p[0] + 0.8 * p[1] * (r - 0.7), x, y); let s = (0.1 * p[1]).exp(); (s * u, s * w) }
            else if f == 10 { let (r, th) = ((x * x + y * y).sqrt(), y.atan2(x)); let r2 = r * (1.0 + 0.15 * p[1] * (3.0 * th).sin()); let t2 = th + 0.5 * p[0]; (r2 * t2.cos(), r2 * t2.sin()) }
            else { let u = x + 0.3 * p[0] * (2.0 * y).tanh(); (u, y + 0.3 * p[1] * (2.0 * u).tanh()) };
        v[2 * i] = a; v[2 * i + 1] = b;
    }
    v
}
fn fin(z: &[f64]) -> bool { z.iter().all(|v| v.is_finite()) }
/// Re-identificación con ÓRBITA como verificador (igual que yo-act): g ∈ 8, p ∈ {p̂} ∪ malla 9×9.
pub fn reid(d: &Orbita, x0: &[f64], ys: &[Vec<f64>], k: usize, phat: [f64; 2]) -> (usize, [f64; 2], f64) {
    let mut ps = vec![phat];
    for i in 0..9 { for j in 0..9 { ps.push([-1.0 + 0.25 * i as f64, -1.0 + 0.25 * j as f64]); } }
    let mut best = (f64::INFINITY, 0usize, [0.0; 2]);
    for g in 0..N_FAM { for p in &ps {
        let cc = context_of(g, *p); let mut z = x0.to_vec(); let mut r = 0.0;
        for t in 1..=k { z = d.step(&z, &cc); r += if fin(&z) { m::rel_err(&z, &ys[t - 1]).min(10.0) } else { 10.0 }; }
        if r < best.0 { best = (r, g, *p); }
    } }
    (best.1, best.2, best.0)
}
pub fn ls_fit(pairs: &[(&[f64], &[f64])]) -> [[f64; 3]; 2] {
    let mut a = [[0.0; 3]; 3]; let mut b = [[0.0; 3]; 2];
    for (x, y) in pairs { for i in 0..x.len() / 2 { let u = [x[2 * i], x[2 * i + 1], 1.0];
        for r in 0..3 { for c in 0..3 { a[r][c] += u[r] * u[c]; } for o in 0..2 { b[o][r] += u[r] * y[2 * i + o]; } } } }
    for r in 0..3 { a[r][r] += 1e-9; }
    let mut out = [[0.0; 3]; 2];
    for o in 0..2 { let mut mm = a; let mut v = b[o];
        for c in 0..3 { let piv = (c..3).max_by(|&i, &j| mm[i][c].abs().total_cmp(&mm[j][c].abs())).unwrap(); mm.swap(c, piv); v.swap(c, piv);
            for r in 0..3 { if r != c { let f = mm[r][c] / mm[c][c]; for cc in 0..3 { mm[r][cc] -= f * mm[c][cc]; } v[r] -= f * v[c]; } } }
        for c in 0..3 { out[o][c] = v[c] / mm[c][c]; } }
    out
}
pub fn ls_step(w: &[[f64; 3]; 2], x: &[f64]) -> Vec<f64> { let mut y = x.to_vec(); for i in 0..x.len() / 2 { for o in 0..2 { y[2 * i + o] = w[o][0] * x[2 * i] + w[o][1] * x[2 * i + 1] + w[o][2]; } } y }
/// fam-set [--rng HEX]: conjunto fijo e independiente de la seed (RNG 0xFA4E por defecto): 16 ejemplos por familia 0–9.
pub fn set(a: &[String]) {
    let r = super::arg(a, "--rng").map(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()).unwrap_or(0xFA4E);
    let mut rng = Xoshiro256StarStar::seed_from_u64(r);
    println!("idx,fam,p1,p2,x0");
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let n: usize = super::arg(a, "--n").map(|s| s.parse().unwrap()).unwrap_or(16);
    let nf: usize = super::arg(a, "--nf").map(|s| s.parse().unwrap()).unwrap_or(NF);
    for f in 0..nf { for i in 0..n {
        let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
        let x = make_object(&mut rng, 6, cls[i % 3]);
        println!("{},{f},{:.17},{:.17},{}", f * n + i, p[0], p[1], x.iter().map(|v| format!("{v:.17}")).collect::<Vec<_>>().join(";"));
    } }
}
fn nums(s: &str, sep: char) -> Vec<f64> { s.split(sep).filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect() }
pub fn eval(a: &[String]) {
    let d = Orbita::from_json(&std::fs::read_to_string(super::arg(a, "--weights").unwrap()).unwrap());
    let set: Vec<(usize, usize, [f64; 2], Vec<f64>)> = std::fs::read_to_string(super::arg(a, "--set").unwrap()).unwrap().lines().skip(1).map(|l| {
        let f: Vec<&str> = l.split(',').collect(); (f[0].parse().unwrap(), f[1].parse().unwrap(), [f[2].parse().unwrap(), f[3].parse().unwrap()], nums(f[4], ';')) }).collect();
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    println!("idx,text_id,fam,base_ok8,base_ok16,obs1_ok8,obs1_ok16,obs1_fam,obs2_ok8,obs2_ok16,obs2_fam,ls1_ok8,ls1_ok16,ls2_ok8,ls2_ok16,true_ok8,true_ok16");
    for line in std::fs::read_to_string(super::arg(a, "--ctx").unwrap()).unwrap().lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let idx: usize = f[0].parse().unwrap();
        let c: Vec<f64> = f[3..13].iter().map(|v| v.parse().unwrap()).collect();
        let (_, fam, p, x0) = set.iter().find(|e| e.0 == idx).unwrap();
        let mut ys = vec![]; let mut y = x0.clone(); for _ in 0..16 { y = rule(*fam, *p, &y); ys.push(y.clone()); }
        let ok = |z: &Vec<f64>, t: usize| fin(z) && m::correct(z, &ys[t - 1]);
        let roll = |start: &Vec<f64>, t0: usize, step: &dyn Fn(&[f64]) -> Vec<f64>| { let mut z = start.clone(); let (mut o8, mut o16) = (false, false);
            for t in t0 + 1..=16 { z = step(&z); if t == 8 { o8 = ok(&z, 8); } if t == 16 { o16 = ok(&z, 16); } } (o8 as u8, o16 as u8) };
        let (b8, b16) = roll(x0, 0, &|z| d.step(z, &c));
        let mut row = format!("{idx},{},{fam},{b8},{b16}", f[2]);
        for k in [1usize, 2] { let (g, pp, _) = reid(&d, x0, &ys, k, [c[8], c[9]]); let cc = context_of(g, pp);
            let (o8, o16) = roll(&ys[k - 1], k, &|z| d.step(z, &cc)); row += &format!(",{o8},{o16},{g}"); }
        let w1 = ls_fit(&[(x0, &ys[0])]); let (l8, l16) = roll(&ys[0], 1, &|z| ls_step(&w1, z)); row += &format!(",{l8},{l16}");
        let w2 = ls_fit(&[(x0, &ys[0]), (&ys[0], &ys[1])]); let (l8, l16) = roll(&ys[1], 2, &|z| ls_step(&w2, z)); row += &format!(",{l8},{l16}");
        if *fam < N_FAM { let ct = context_of(*fam, *p); let (t8, t16) = roll(x0, 0, &|z| d.step(z, &ct)); row += &format!(",{t8},{t16}"); } else { row += ",,"; }
        println!("{row}");
    }
    eprintln!("queries={}", cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total());
}
/// demo-step --weights W --c "c0..c9" --x0 "12 números" [--obs "y1|y2"]  → JSON
pub fn demo(a: &[String]) {
    let d = Orbita::from_json(&std::fs::read_to_string(super::arg(a, "--weights").unwrap()).unwrap());
    let c = nums(&super::arg(a, "--c").unwrap(), ','); let x0 = nums(&super::arg(a, "--x0").unwrap(), ',');
    let ys: Vec<Vec<f64>> = super::arg(a, "--obs").map(|s| s.split('|').map(|t| nums(t, ',')).collect()).unwrap_or_default();
    let traj = |start: &Vec<f64>, n: usize, cc: &Vec<f64>| { let mut z = start.clone(); (0..n).map(|_| { z = d.step(&z, cc); z.clone() }).collect::<Vec<_>>() };
    let base = traj(&x0, 16, &c);
    let sigma = m::sigma_max(&m::jacobian(|w| d.step(w, &c), &x0));
    let normr = if fin(&base[15]) { m::norm(&base[15]) / m::norm(&x0).max(1e-12) } else { 1e6 };
    let mut out = serde_json::json!({"base_traj": base, "sigma": sigma, "normr": normr});
    if !ys.is_empty() {
        let k = ys.len();
        out["surprise"] = serde_json::json!(m::rel_err(&base[0], &ys[0]));
        let (g, p, r) = reid(&d, &x0, &ys, k, [c[8], c[9]]);
        let mut t: Vec<Vec<f64>> = ys.clone(); t.extend(traj(&ys[k - 1], 16 - k, &context_of(g, p)));
        out["reid"] = serde_json::json!({"fam": g, "p": p, "fit_err": r, "k": k}); out["reid_traj"] = serde_json::json!(t);
    }
    println!("{}", out);
}

/// demo-rule --fam F --p p1,p2 --x "12": aplica la regla verdadera (para simular observaciones en la demo)
pub fn demo_rule(a: &[String]) {
    let f: usize = super::arg(a, "--fam").unwrap().parse().unwrap(); let p = nums(&super::arg(a, "--p").unwrap(), ',');
    let y = rule(f, [p[0], p[1]], &nums(&super::arg(a, "--x").unwrap(), ','));
    println!("{}", y.iter().map(|v| format!("{v:.17}")).collect::<Vec<_>>().join(","));
}
