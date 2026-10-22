//! yo-feat (docs/preregistro_yo.md): rasgos del yo por fila de contexto. --part test (llm_subset, como llm-eval) | dev (ds.dev, 8 rot + 8 refl).
use crate::model::Orbita;
use cdt_rqm_epr::v4_dataset::{apply_rule, Example};
use cdt_rqm_epr::v4_metrics as m;
pub fn dev_subset(s: &cdt_rqm_epr::v4_dataset::Sealed) -> Vec<(usize, &Example)> {
    let (mut r, mut f) = (0, 0); let mut out = vec![];
    for (i, e) in s.ds.dev.iter().enumerate() {
        if e.family_id == 1 && r < 8 { r += 1; out.push((i, e)); }
        if e.family_id == 2 && f < 8 { f += 1; out.push((i, e)); }
    }
    out
}
pub fn run(a: &[String]) {
    let seed = super::seed_of(a); let s = super::sealed(seed);
    let d = Orbita::from_json(&std::fs::read_to_string(super::arg(a, "--weights").unwrap()).unwrap());
    let part = super::arg(a, "--part").unwrap();
    let sub = if part == "dev" { dev_subset(&s) } else { super::llm_subset(&s) };
    if a.iter().any(|x| x == "--dump") { println!("idx,fam,p1,p2"); for (i, e) in &sub { println!("{i},{},{:.17},{:.17}", e.family_id, e.params[0], e.params[1]); } return; }
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    println!("seed,idx,fam,cond,text_id,ok1,ok16,rel16,surprise,sigma,normr");
    for line in std::fs::read_to_string(super::arg(a, "--ctx").unwrap()).unwrap().lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let idx: usize = f[0].parse().unwrap();
        let c: Vec<f64> = f[3..13].iter().map(|v| v.parse().unwrap()).collect();
        let e = sub.iter().find(|(i, _)| *i == idx).expect("idx").1;
        let (mut z, mut y) = (e.input.clone(), e.input.clone());
        let (mut ok1, mut ok16, mut rel16, mut sur) = (false, false, 10.0, 10.0);
        for t in 1..=16usize {
            z = d.step(&z, &c); y = apply_rule(e.family_id, e.params, &y);
            let fin = z.iter().all(|v| v.is_finite());
            if t == 1 { ok1 = fin && m::correct(&z, &y); sur = if fin { m::rel_err(&z, &y).min(10.0) } else { 10.0 }; }
            if t == 16 { ok16 = fin && m::correct(&z, &y); rel16 = if fin { m::rel_err(&z, &y).min(10.0) } else { 10.0 }; }
        }
        let sig = m::sigma_max(&m::jacobian(|w| d.step(w, &c), &e.input));
        let nr = if z.iter().all(|v| v.is_finite()) { m::norm(&z) / m::norm(&e.input).max(1e-12) } else { 1e6 };
        println!("0x{seed:X},{idx},{},{},{},{},{},{rel16:.6},{sur:.8},{sig:.6},{nr:.6}", e.family_id, f[1], f[2], ok1 as u8, ok16 as u8);
    }
    eprintln!("0x{seed:X} part={part} queries={}", cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total());
}

/// yo-act (docs/preregistro_yo_act.md): por fila, BASE (ĉ desde x0) y OBSk (k=1,2): se observan y1..yk reales, se re-identifica (familia, p)
/// con ÓRBITA como verificador (min Σ_t≤k rel_err(ÓRBITA^t(x0, c(g,p)), y_t); p ∈ {p̂} ∪ malla 9×9) y se sigue el rollout desde y_k.
pub fn act(a: &[String]) {
    let seed = super::seed_of(a); let s = super::sealed(seed);
    let d = Orbita::from_json(&std::fs::read_to_string(super::arg(a, "--weights").unwrap()).unwrap());
    let part = super::arg(a, "--part").unwrap();
    let sub = if part == "dev" { dev_subset(&s) } else { super::llm_subset(&s) };
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let ctxof = |g: usize, p: [f64; 2]| cdt_rqm_epr::v4_dataset::context_of(g, p);
    println!("idx,text_id,base_ok8,base_ok16,obs1_ok8,obs1_ok16,obs1_fam,obs2_ok8,obs2_ok16,obs2_fam");
    for line in std::fs::read_to_string(super::arg(a, "--ctx").unwrap()).unwrap().lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let idx: usize = f[0].parse().unwrap();
        let c: Vec<f64> = f[3..13].iter().map(|v| v.parse().unwrap()).collect();
        let e = sub.iter().find(|(i, _)| *i == idx).expect("idx").1;
        let mut ys = vec![e.input.clone()];
        for t in 0..16 { let y = apply_rule(e.family_id, e.params, &ys[t]); ys.push(y); }
        let ok = |z: &Vec<f64>, t: usize| z.iter().all(|v| v.is_finite()) && m::correct(z, &ys[t]);
        let roll = |start: &Vec<f64>, t0: usize, cc: &Vec<f64>| -> (bool, bool) {
            let mut z = start.clone(); let (mut o8, mut o16) = (false, false);
            for t in t0 + 1..=16 { z = d.step(&z, cc); if t == 8 { o8 = ok(&z, 8); } if t == 16 { o16 = ok(&z, 16); } }
            (o8, o16)
        };
        let (b8, b16) = roll(&e.input, 0, &c);
        let mut row = format!("{idx},{},{},{}", f[2], b8 as u8, b16 as u8);
        let mut ps = vec![[c[8], c[9]]];
        for i in 0..9 { for j in 0..9 { ps.push([-1.0 + 0.25 * i as f64, -1.0 + 0.25 * j as f64]); } }
        for k in [1usize, 2] {
            let mut best = (f64::INFINITY, 0usize, [0.0; 2]);
            for g in 0..cdt_rqm_epr::v4_dataset::N_FAM { for p in &ps {
                let cc = ctxof(g, *p); let mut z = e.input.clone(); let mut r = 0.0;
                for t in 1..=k { z = d.step(&z, &cc); r += if z.iter().all(|v| v.is_finite()) { m::rel_err(&z, &ys[t]).min(10.0) } else { 10.0 }; }
                if r < best.0 { best = (r, g, *p); }
            } }
            let (o8, o16) = roll(&ys[k], k, &ctxof(best.1, best.2));
            row += &format!(",{},{},{}", o8 as u8, o16 as u8, best.1);
        }
        println!("{row}");
    }
    eprintln!("0x{seed:X} part={part} queries={}", cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total());
}
