//! Benchmark pareado de las dos rutas del ratio Jastrow.
//!
//! Fuerza barrido contiguo e incidencias sobre la misma geometría, semilla y
//! presupuesto. Permite calibrar el punto de cruce de la estrategia híbrida.

use cdt_rqm_epr::quantum_spin_thermodynamic_engine::periodic_pyrochlore_model;
use cdt_rqm_epr::symmetry_thermodynamic_substrate::SymmetryThermodynamicConfig;
use cdt_rqm_epr::variational_spin_liquid_vmc::{
    ComplexJastrowVmc, VmcRatioStrategy, VmcSpinConfig,
};
use std::hint::black_box;
use std::time::Instant;

const REPS: usize = 5;

fn dimensions(spins: usize) -> (usize, usize, usize) {
    match spins {
        8 => (2, 1, 1),
        16 => (2, 2, 1),
        32 => (2, 2, 2),
        64 => (4, 2, 2),
        _ => panic!("fixture VMC no soportada: {spins}"),
    }
}

fn run_strategy(
    spins: usize,
    samples: usize,
    incidence_ratio_min_spins: usize,
) -> (f64, f64, VmcRatioStrategy) {
    let (nx, ny, nz) = dimensions(spins);
    let (geometry, bonds) =
        periodic_pyrochlore_model(nx, ny, nz, SymmetryThermodynamicConfig::default())
            .expect("geometría pyrochlore válida");
    let mut timings = Vec::with_capacity(REPS);
    let mut energy = 0.0;
    let mut strategy = VmcRatioStrategy::ContiguousScan;
    for _ in 0..REPS {
        let mut vmc = ComplexJastrowVmc::new_with_bonds(
            geometry.clone(),
            bonds.clone(),
            VmcSpinConfig {
                incidence_ratio_min_spins,
                ..VmcSpinConfig::default()
            },
        );
        strategy = vmc.ratio_strategy();
        let started = Instant::now();
        let report = black_box(vmc.sample_report(samples, 50, 1));
        timings.push(started.elapsed().as_secs_f64() * 1_000.0);
        energy = report.energy;
    }
    timings.sort_by(f64::total_cmp);
    (timings[timings.len() / 2], energy, strategy)
}

fn main() {
    println!("spins,samples,contiguous_ms,incidence_ms,default_strategy,best_speedup,energy_drift");
    for (spins, samples) in [(8, 20_000), (16, 4_000), (32, 1_000), (64, 250)] {
        let (contiguous_ms, contiguous_energy, _) = run_strategy(spins, samples, usize::MAX);
        let (incidence_ms, incidence_energy, _) = run_strategy(spins, samples, 0);
        let default_strategy = if spins < VmcSpinConfig::default().incidence_ratio_min_spins {
            VmcRatioStrategy::ContiguousScan
        } else {
            VmcRatioStrategy::PairIncidence
        };
        let best_speedup = contiguous_ms.max(incidence_ms) / contiguous_ms.min(incidence_ms);
        println!(
            "{spins},{samples},{contiguous_ms:.3},{incidence_ms:.3},{default_strategy:?},\
             {best_speedup:.3},{:.3e}",
            (contiguous_energy - incidence_energy).abs()
        );
    }
}
