//! Puente H2H token-free vs v4 (se copia a `examples/` de un worktree
//! desacoplado de `exp/field-autonomy-v4` @1da52e7; NO se commitea en v4).
//! Por semilla: exporta el dataset sellado (TRAIN/TEST) + composiciones
//! deterministas, y mide v4 Dφ A/B (E35 core) en h1, rollout (familias
//! iterables) y composición 2-pasos, tiempos de entrenamiento y latencia.
//! Uso: cargo run --release --example h2h_v4 -- dev|confirm <out.json>
use cdt_rqm_epr::field_autonomy_v4::{
    run_e35_core, target_h, CONFIRM_SEEDS_V4, DEV_SEEDS_V4, HORIZONS, ITERABLE_FAMS, K_POINTS,
};
use cdt_rqm_epr::v4_controls::Predictor;
use cdt_rqm_epr::v4_dataset::{apply_rule, generate_and_seal, DatasetConfig, Example};
use cdt_rqm_epr::v4_metrics as m;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde_json::json;
use std::time::Instant;

fn ex(e: &Example) -> serde_json::Value {
    json!({"fam": e.family_id, "p": e.params, "x": e.input, "c": e.context, "y": e.expected})
}

fn acc_roll(p: &dyn Predictor, xs: &[&Example], h: usize) -> f64 {
    let ok = xs
        .iter()
        .filter(|e| {
            let mut z = e.input.clone();
            for _ in 0..h {
                z = p.predict(&z, &e.context);
            }
            m::correct(&z, &target_h(e, h))
        })
        .count();
    ok as f64 / xs.len().max(1) as f64
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let seeds: Vec<u64> = if a[1] == "dev" {
        DEV_SEEDS_V4.to_vec()
    } else {
        CONFIRM_SEEDS_V4.to_vec()
    };
    let cfg = DatasetConfig::standard(K_POINTS);
    let mut out = vec![];
    for seed in seeds {
        let s = generate_and_seal(seed, &cfg);
        let test = &s.ds.test;
        // Composición 2 pasos determinista: (e_i, contexto de e_j).
        let mut r = Xoshiro256StarStar::seed_from_u64(seed ^ 0xC0B0);
        let comp: Vec<(usize, usize)> = (0..test.len())
            .map(|i| (i, r.gen_range(0..test.len())))
            .collect();
        let t0 = Instant::now();
        let o = run_e35_core(&s, seed, K_POINTS, "H2H", false);
        let t_e35 = t0.elapsed().as_secs_f64();
        let iter: Vec<&Example> = test
            .iter()
            .filter(|e| ITERABLE_FAMS.contains(&e.family_id))
            .collect();
        let mut res = serde_json::Map::new();
        for (name, d) in [("A", &o.a as &dyn Predictor), ("B", &o.b as &dyn Predictor)] {
            let roll: Vec<f64> = HORIZONS.iter().map(|&h| acc_roll(d, &iter, h)).collect();
            let c_ok = comp
                .iter()
                .filter(|&&(i, j)| {
                    let (ei, ej) = (&test[i], &test[j]);
                    let z = d.predict(&d.predict(&ei.input, &ei.context), &ej.context);
                    let y = apply_rule(ej.family_id, ej.params, &ei.expected);
                    m::correct(&z, &y)
                })
                .count() as f64
                / comp.len() as f64;
            // Latencia: 1 predicción (objeto completo, K=6 puntos), 20 repeticiones.
            let mut lat = vec![];
            for _ in 0..20 {
                for e in test {
                    let t = Instant::now();
                    std::hint::black_box(d.predict(&e.input, &e.context));
                    lat.push(t.elapsed().as_nanos() as f64 / 1000.0);
                }
            }
            lat.sort_by(|x, y| x.total_cmp(y));
            let q = |f: f64| lat[((lat.len() - 1) as f64 * f) as usize];
            res.insert(
                name.into(),
                json!({"acc_h1": if name == "A" {o.ev_a.acc} else {o.ev_b.acc},
                    "roll_iterable": roll, "comp2": c_ok, "params": d.params(),
                    "lat_us": {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99)}}),
            );
        }
        let q = o.ev_b.queries.total();
        out.push(json!({"seed": seed, "train": s.ds.train.iter().map(ex).collect::<Vec<_>>(),
            "test": test.iter().map(ex).collect::<Vec<_>>(), "comp": comp,
            "v4": res, "e35_core_wall_s": t_e35, "test_memory_queries_B": q,
            "leaked_B": o.ev_b.leaked, "manifest_config_hash": s.manifest.config_hash}));
        eprintln!("seed {seed:#x} hecho en {t_e35:.1}s");
    }
    std::fs::write(&a[2], serde_json::to_string(&out).unwrap()).unwrap();
}
