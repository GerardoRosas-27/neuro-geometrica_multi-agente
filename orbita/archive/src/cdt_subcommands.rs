//! ARCHIVADO (no compilado). Subcomandos CDT retirados de la ruta activa en exp/orbita-yo; extraídos verbatim de orbita/src/main.rs @ e2f6bb0 (líneas 367–528).
/// Rasgos CDT por familia candidata (prerregistro docs/preregistro_cdt_sueno.md). Para cada fila (texto) y familia g:
/// r_g = rel_err(ÓRBITA h16 con c_g(p), teacher consolidado h16 con g) [10 si el teacher no tiene la familia],
/// sup_g = p dentro del rango de la regla, tr_g = familia de confianza (residuo de ajuste < TRUST_TAU).
/// --signal full | corrupt | sleep. sleep = consolidar → máscara de confianza (poda de reglas no confiables) →
/// borrar la CDT episódica → replay generativo (pseudo_experiences) → re-consolidar offline sobre el replay.
fn cdt_feat(a: &[String]) {
    let seed = seed_of(a);
    let s = sealed(seed);
    let d = Orbita::from_json(&std::fs::read_to_string(arg(a, "--weights").expect("--weights")).unwrap());
    let mode = arg(a, "--signal").unwrap_or("full".into());
    let sub = llm_subset(&s);
    let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
    let full = consolidate(&cdt, if mode == "corrupt" { ConsolidateMode::Corrupt } else { ConsolidateMode::Full });
    let res = family_fit_residual(&full, cdt.read_all());
    let n_exp = cdt.read_all().len();
    drop(cdt);
    let trusted: Vec<bool> = (0..N_FAM).map(|f| res.get(f).map_or(false, |r| *r < TRUST_TAU)).collect();
    let n_full = full.n_numbers();
    let sig = if mode == "sleep" {
        let pruned = LearningSignal { kind: full.kind.clone(), rules: full.rules.iter().filter(|r| trusted[r.fam]).cloned().collect(), generic: None };
        let pseudo = pseudo_experiences(&pruned, s.ds.train.len(), K_POINTS, seed ^ 0x5EE9);
        let c2 = collect_experience(&pseudo, ExpMode::Real, seed ^ 0x5EE9);
        consolidate(&c2, ConsolidateMode::Full)
    } else { full };
    let before = snapshot(); // como llm-eval: se cuentan las consultas de la fase de inferencia
    let fams_in: Vec<usize> = sig.rules.iter().map(|r| r.fam).collect();
    let txt = std::fs::read_to_string(arg(a, "--rows").expect("--rows")).unwrap();
    let mut hdr = "id".to_string();
    for g in 0..N_FAM { hdr += &format!(",r{g},sup{g},tr{g}"); }
    println!("{hdr}");
    for line in txt.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let p = [f[3].parse::<f64>().unwrap(), f[4].parse::<f64>().unwrap()];
        let x: Vec<f64> = if f[1] == "test" {
            let idx: usize = f[2].parse().unwrap();
            sub.iter().find(|(i, _)| *i == idx).expect("idx").1.input.clone()
        } else {
            let fam: usize = f[2].parse().unwrap();
            let pool: Vec<&Example> = s.ds.train.iter().filter(|e| e.family_id == fam).collect();
            let h = f[0].bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
            pool[(h % pool.len() as u64) as usize].input.clone()
        };
        let mut row = f[0].to_string();
        for g in 0..N_FAM {
            let mut c = vec![0.0; N_FAM + 2]; c[g] = 1.0; c[N_FAM] = p[0]; c[N_FAM + 1] = p[1];
            let (mut z, mut t) = (x.clone(), Some(x.clone()));
            for _ in 0..16 { z = d.step(&z, &c); t = t.and_then(|v| sig.predict(g, p, &v)); }
            let r = match t { Some(t) if z.iter().all(|v| v.is_finite()) => m::rel_err(&z, &t).min(10.0), _ => 10.0 };
            let sup = sig.rules.iter().find(|q| q.fam == g).map_or(0, |q| (p[0] >= q.p_min[0] && p[0] <= q.p_max[0] && p[1] >= q.p_min[1] && p[1] <= q.p_max[1]) as u8);
            row += &format!(",{r:.6},{sup},{}", (trusted[g] && fams_in.contains(&g)) as u8);
        }
        println!("{row}");
    }
    eprintln!("0x{seed:X} signal={mode} episodes={n_exp} numbers_full={n_full} numbers_used={} rules_used={} trusted={:?} queries={}",
        sig.n_numbers(), sig.rules.len(), trusted, snapshot().delta(&before).total());
}

/// Señal consolidada según modo (mismos mecanismos v4 que cdt-feat): full | corrupt | sleep (máscara trusted → poda →
/// borrar CDT → pseudo_experiences → re-consolidar). Devuelve (señal, números antes, números después).
fn build_signal(s: &Sealed, seed: u64, mode: &str) -> (LearningSignal, usize, usize) {
    let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
    let full = consolidate(&cdt, if mode == "corrupt" { ConsolidateMode::Corrupt } else { ConsolidateMode::Full });
    let res = family_fit_residual(&full, cdt.read_all());
    drop(cdt);
    let n0 = full.n_numbers();
    if mode != "sleep" { return (full, n0, n0); }
    let trusted: Vec<bool> = (0..N_FAM).map(|f| res.get(f).map_or(false, |r| *r < TRUST_TAU)).collect();
    let pruned = LearningSignal { kind: full.kind.clone(), rules: full.rules.iter().filter(|r| trusted[r.fam]).cloned().collect(), generic: None };
    let pseudo = pseudo_experiences(&pruned, s.ds.train.len(), K_POINTS, seed ^ 0x5EE9);
    let sig = consolidate(&collect_experience(&pseudo, ExpMode::Real, seed ^ 0x5EE9), ConsolidateMode::Full);
    let n1 = sig.n_numbers();
    (sig, n0, n1)
}

/// Pruebas causales (docs/preregistro_cdt_causal.md). Para cada fila E2E (familia D2 + p de numnorm) y su ejemplo TEST
/// iterable: se observan los K_OBS=2 primeros estados reales (online, inferencia; nunca se entrena con ellos).
/// Salida por (fila, condición): ok1, ok8, ok16 (vs trayectoria real desde x0) y score (para AUROC de fallos).
fn cdt_causal(a: &[String]) {
    const K_OBS: usize = 2;
    const ALPHA: f64 = 0.5;
    let seed = seed_of(a);
    let s = sealed(seed);
    let d = Orbita::from_json(&std::fs::read_to_string(arg(a, "--weights").expect("--weights")).unwrap());
    let sub = llm_subset(&s);
    let sigs: Vec<(&str, LearningSignal, usize, usize)> = ["full", "sleep", "corrupt"].iter().map(|m| { let (g, n0, n1) = build_signal(&s, seed, m); (*m, g, n0, n1) }).collect();
    let before = snapshot();
    let txt = std::fs::read_to_string(arg(a, "--ctx").expect("--ctx")).unwrap();
    let rows: Vec<(usize, String, Vec<f64>)> = txt.lines().skip(1).filter(|l| l.split(',').nth(1) == Some("E2E_R"))
        .map(|l| { let f: Vec<&str> = l.split(',').collect(); (f[0].parse().unwrap(), f[2].to_string(), f[3..13].iter().map(|v| v.parse().unwrap()).collect()) }).collect();
    let n = rows.len();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xCA5A);
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() { let j = rng.gen_range(0..=i); perm.swap(i, j); }
    let der: Vec<usize> = { let o = rng.gen_range(1..N_FAM); (0..N_FAM).map(|g| (g + o) % N_FAM).collect() }; // desarreglo fijo de familias
    let cg = |g: usize, p: [f64; 2]| { let mut c = vec![0.0; N_FAM + 2]; c[g] = 1.0; c[N_FAM] = p[0]; c[N_FAM + 1] = p[1]; c };
    // trayectorias reales y observaciones
    let ex: Vec<(&Example, Vec<Vec<f64>>)> = rows.iter().map(|(idx, _, _)| {
        let e = sub.iter().find(|(i, _)| i == idx).expect("idx").1;
        let mut y = vec![e.input.clone()];
        for t in 0..16 { let v = cdt_rqm_epr::v4_dataset::apply_rule(e.family_id, e.params, &y[t]); y.push(v); }
        (e, y)
    }).collect();
    // residuo por familia sobre las K_OBS observaciones: predictor f(g, x) -> Option<x'>
    let obs_res = |y: &Vec<Vec<f64>>, f: &dyn Fn(usize, &[f64]) -> Option<Vec<f64>>| -> Vec<f64> {
        (0..N_FAM).map(|g| { let mut r = 0.0; for t in 0..K_OBS { match f(g, &y[t]) { Some(q) if q.iter().all(|v| v.is_finite()) => r += m::rel_err(&q, &y[t + 1]).min(10.0), _ => r += 10.0 } } r }).collect()
    };
    let argmin = |v: &[f64]| (0..v.len()).min_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
    // rollout: corrector opcional (señal, familia del teacher, alpha)
    let roll = |x0: &[f64], c: &[f64], cor: Option<(&LearningSignal, usize, [f64; 2])>| -> Vec<Vec<f64>> {
        let mut z = x0.to_vec(); let mut out = vec![];
        for _ in 0..16 {
            let o = d.step(&z, c);
            z = match cor { Some((sg, g, p)) => match sg.predict(g, p, &z) { Some(t) => o.iter().zip(&t).map(|(u, v)| (1.0 - ALPHA) * u + ALPHA * v).collect(), None => o }, None => o };
            out.push(z.clone());
        }
        out
    };
    // ajuste afín por mínimos cuadrados sobre las K_OBS observaciones (oráculo-justo sin CDT ni ÓRBITA)
    let ls = |y: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        let (mut xs, mut tx, mut ty) = (vec![], vec![], vec![]);
        for t in 0..K_OBS { for i in 0..y[t].len() / 2 { xs.push(vec![y[t][2 * i], y[t][2 * i + 1], 1.0]); tx.push(vec![y[t + 1][2 * i]]); ty.push(vec![y[t + 1][2 * i + 1]]); } }
        let wx = cdt_rqm_epr::v4_controls::ridge(&xs, &tx, 1e-6); let wy = cdt_rqm_epr::v4_controls::ridge(&xs, &ty, 1e-6);
        let mut z = y[0].clone(); let mut out = vec![];
        for _ in 0..16 { let mut q = z.clone(); for i in 0..z.len() / 2 { let v = [z[2 * i], z[2 * i + 1], 1.0]; q[2 * i] = (0..3).map(|k| v[k] * wx[k][0]).sum(); q[2 * i + 1] = (0..3).map(|k| v[k] * wy[k][0]).sum(); } z = q; out.push(z.clone()); }
        out
    };
    let dr = &d;
    // residuos por fila para OBS (full / sleep / corrupt / ÓRBITA)
    let mut res: std::collections::HashMap<&str, Vec<Vec<f64>>> = Default::default();
    for (nm, sg, _, _) in &sigs { res.insert(nm, (0..n).map(|i| { let p = [rows[i].2[8], rows[i].2[9]]; obs_res(&ex[i].1, &|g, x| sg.predict(g, p, x)) }).collect()); }
    res.insert("orb", (0..n).map(|i| { let p = [rows[i].2[8], rows[i].2[9]]; obs_res(&ex[i].1, &|g, x| Some(dr.step(x, &cg(g, p)))) }).collect());
    println!("idx,text_id,fam,cond,ok1,ok8,ok16,score");
    for i in 0..n {
        let (idx, tid, c) = &rows[i]; let (e, y) = &ex[i]; let p = [c[8], c[9]]; let (fh, _) = fam_of(c);
        let mut put = |cond: &str, zs: &Vec<Vec<f64>>, score: f64| {
            let ok = |h: usize| (zs[h - 1].iter().all(|v| v.is_finite()) && m::correct(&zs[h - 1], &y[h])) as u8;
            println!("{idx},{tid},{},{cond},{},{},{},{score:.6}", e.family_id, ok(1), ok(8), ok(16));
        };
        let base = roll(&e.input, c, None); let rb = res["orb"][i][fh];
        put("BASE", &base, rb);
        let sel = |v: &Vec<f64>| argmin(v);
        for (tag, key) in [("DIR", "full"), ("SLEEP", "sleep"), ("CORRUPT", "corrupt")] {
            let sig = &sigs.iter().find(|q| q.0 == key).unwrap().1;
            let r = &res[key][i]; let g = sel(r);
            put(&format!("OBS_{tag}"), &roll(&e.input, &cg(g, p), None), r[g]);
            put(&format!("CORR_{tag}"), &roll(&e.input, c, Some((sig, fh, p))), res[key][i][fh]);
            put(&format!("OBSCORR_{tag}"), &roll(&e.input, &cg(g, p), Some((sig, g, p))), r[g]);
            if tag == "DIR" {
                let rs = &res[key][perm[i]]; let gs = sel(rs);   // SHUF: residuos de otra fila (observaciones ajenas)
                put("OBS_SHUF", &roll(&e.input, &cg(gs, p), None), rs[gs]);
                put("CORR_SHUF", &roll(&e.input, c, Some((sig, der[fh], p))), r[der[fh]]);
                put("OBSCORR_SHUF", &roll(&e.input, &cg(gs, p), Some((sig, der[gs], p))), rs[gs]);
            }
        }
        let ro = &res["orb"][i]; let go = sel(ro);
        put("ORACLE_ORB", &roll(&e.input, &cg(go, p), None), ro[go]);
        let l = ls(y); let lsr = { let mut r = 0.0; let mut z = y[0].clone(); for t in 0..K_OBS { let _ = t; z = l[t].clone(); r += m::rel_err(&z, &y[t + 1]).min(10.0); } r };
        put("ORACLE_LS", &l, lsr);
    }
    let sz: Vec<String> = sigs.iter().map(|(nm, _, n0, n1)| format!("{nm}:{n0}->{n1}")).collect();
    eprintln!("0x{seed:X} sizes={} queries={}", sz.join(","), snapshot().delta(&before).total());
}
