//! Sustrato de campo sin tokens.
//!
//! El estado no es una secuencia de IDs. Es
//! `Ψ = (T, z, m, C)`: complejo simplicial, fasores en aristas,
//! máscara de consolidación y cavidades Casimir.
//!
//! Inferencia = relajar `z` (Langevin + atención geométrica).
//! Aprendizaje = Hebb local sobre el atractor, no next-token.

#![allow(clippy::needless_range_loop)] // índices de aristas/caras, no iteradores de colección

use num_complex::Complex;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

pub type Phasor = Complex<f64>;

const EPS: f64 = 1.0e-12;

/// Complejo simplicial 2D orientado (vértices, aristas, caras).
#[derive(Clone, Debug)]
pub struct ComplexT {
    pub n_vertices: usize,
    pub edges: Vec<(usize, usize)>,
    pub faces: Vec<[usize; 3]>,
}

impl ComplexT {
    /// Octaedro: 6 vértices, 12 aristas, 8 caras triangulares.
    pub fn octahedron() -> Self {
        Self {
            n_vertices: 6,
            edges: vec![
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 2),
            ],
            faces: vec![
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 5],
                [0, 5, 2],
                [1, 3, 2],
                [1, 4, 3],
                [1, 5, 4],
                [1, 2, 5],
            ],
        }
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn edge_index(&self, a: usize, b: usize) -> Option<usize> {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        self.edges.iter().position(|&e| e == (a, b))
    }

    /// Peso geométrico: 0 si no comparten vértice; 1 si comparten frontera.
    pub fn boundary_weight(&self, e: usize, e2: usize) -> f64 {
        if e == e2 {
            return 1.0;
        }
        let (a, b) = self.edges[e];
        let (c, d) = self.edges[e2];
        let share = a == c || a == d || b == c || b == d;
        if share {
            1.0
        } else {
            0.0
        }
    }

    pub fn share_boundary(&self, e: usize, e2: usize) -> bool {
        self.boundary_weight(e, e2) > 0.0
    }

    /// Aristas-puente entre dos caras (tocan ambas y no pertenecen a ninguna).
    pub fn bridge_edges(&self, face_a: usize, face_b: usize) -> Vec<usize> {
        let fa = self.faces[face_a];
        let fb = self.faces[face_b];
        let in_face = |face: [usize; 3], e: (usize, usize)| -> bool {
            let (x, y) = e;
            face.contains(&x) && face.contains(&y)
        };
        let touches = |face: [usize; 3], e: (usize, usize)| -> bool {
            face.contains(&e.0) || face.contains(&e.1)
        };
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, &e)| {
                !in_face(fa, e) && !in_face(fb, e) && touches(fa, e) && touches(fb, e)
            })
            .map(|(i, _)| i)
            .collect()
    }
}

/// Estado de campo. Cero vocabulario.
#[derive(Clone, Debug)]
pub struct FieldState {
    pub t: ComplexT,
    pub z: Vec<Phasor>,
    pub z_past: Vec<Phasor>,
    pub m: Vec<f64>,
    pub casimir: Vec<(usize, usize)>,
    pub l1: Vec<f64>,
}

impl FieldState {
    pub fn new(t: ComplexT) -> Self {
        let n = t.n_edges();
        let mut l1 = vec![0.0; n * n];
        for e in 0..n {
            let mut deg = 0.0;
            for e2 in 0..n {
                if e != e2 && t.share_boundary(e, e2) {
                    l1[e * n + e2] = -1.0;
                    deg += 1.0;
                }
            }
            l1[e * n + e] = deg;
        }
        Self {
            z: vec![Phasor::new(0.0, 0.0); n],
            z_past: vec![Phasor::new(0.0, 0.0); n],
            m: vec![0.0; n],
            casimir: Vec::new(),
            l1,
            t,
        }
    }

    pub fn n(&self) -> usize {
        self.t.n_edges()
    }

    fn l1_at(&self, e: usize, e2: usize) -> f64 {
        self.l1[e * self.n() + e2]
    }

    fn l1_at_mut(&mut self, e: usize, e2: usize) -> &mut f64 {
        let n = self.n();
        &mut self.l1[e * n + e2]
    }

    /// No hay tokens: el estado no guarda IDs de vocabulario.
    pub fn has_token_ids(&self) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FieldConfig {
    pub dt: f64,
    pub temperature: f64,
    pub lambda: f64,
    pub casimir_kappa: f64,
    pub handshake_threshold: f64,
    pub rigid_amp: f64,
    pub hebb_eta: f64,
    pub prune_eps: f64,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            dt: 0.05,
            temperature: 0.002,
            lambda: 0.05,
            casimir_kappa: 0.15,
            handshake_threshold: 0.80,
            rigid_amp: 0.25,
            hebb_eta: 0.08,
            prune_eps: 0.02,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EnergyReport {
    pub hopfield: f64,
    pub hodge: f64,
    pub regularizer: f64,
    pub casimir: f64,
    pub total: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CycleReport {
    pub energy_before: f64,
    pub energy_after: f64,
    pub handshake: f64,
    pub rigid_edges: usize,
    pub casimir_cavities: usize,
    pub live_edges: usize,
    pub recon_error: f64,
    pub baseline_error: f64,
}

pub fn hopfield_handshake(z_past: &[Phasor], z: &[Phasor]) -> f64 {
    let mut num = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (a, b) in z_past.iter().zip(z.iter()) {
        num += (a.conj() * b).re;
        na += a.norm_sqr();
        nb += b.norm_sqr();
    }
    num / (na.sqrt() * nb.sqrt() + EPS)
}

pub fn free_energy(psi: &FieldState, cfg: &FieldConfig) -> EnergyReport {
    let n = psi.n();
    let mut hopfield = 0.0;
    let mut regularizer = 0.0;
    for e in 0..n {
        hopfield -= (psi.z_past[e].conj() * psi.z[e]).re;
        regularizer += cfg.lambda * psi.z[e].norm_sqr();
    }
    let mut hodge = 0.0;
    for e in 0..n {
        for e2 in 0..n {
            hodge += psi.l1_at(e, e2) * (psi.z[e].conj() * psi.z[e2]).re;
        }
    }
    let mut casimir = 0.0;
    for &(a, b) in &psi.casimir {
        for e in psi.t.bridge_edges(a, b) {
            casimir += cfg.casimir_kappa * psi.z[e].norm_sqr();
        }
    }
    EnergyReport {
        hopfield,
        hodge,
        regularizer,
        casimir,
        total: hopfield + hodge + regularizer + casimir,
    }
}

/// Afinidad geométrica: solape de fasores que comparten frontera.
pub fn geometric_affinity(psi: &FieldState, e: usize, e2: usize) -> f64 {
    let w = psi.t.boundary_weight(e, e2);
    if w <= 0.0 {
        return 0.0;
    }
    (psi.z[e].conj() * psi.z[e2]).re * w
}

fn attention_mix(psi: &FieldState, e: usize) -> Phasor {
    let n = psi.n();
    let mut local_max = f64::NEG_INFINITY;
    for e2 in 0..n {
        if psi.t.boundary_weight(e, e2) <= 0.0 {
            continue;
        }
        let a = geometric_affinity(psi, e, e2);
        if a > local_max {
            local_max = a;
        }
    }
    if !local_max.is_finite() {
        return Phasor::new(0.0, 0.0);
    }
    let mut acc = Phasor::new(0.0, 0.0);
    let mut zsum = 0.0;
    for e2 in 0..n {
        if psi.t.boundary_weight(e, e2) <= 0.0 {
            continue;
        }
        let w = (geometric_affinity(psi, e, e2) - local_max).exp();
        acc += psi.z[e2] * w;
        zsum += w;
    }
    if zsum > EPS {
        acc / zsum
    } else {
        Phasor::new(0.0, 0.0)
    }
}

fn energy_gradient(psi: &FieldState, cfg: &FieldConfig, e: usize) -> Phasor {
    let n = psi.n();
    // dF_hop / d conj(z_e) ~ -z_past / 2, but we step in real coords:
    // F_hop = -Re(z_past* z) → grad_x = -Re(z_past), grad_y = -Im(z_past)
    let hop = -psi.z_past[e];
    let reg = Phasor::new(
        2.0 * cfg.lambda * psi.z[e].re,
        2.0 * cfg.lambda * psi.z[e].im,
    );
    let mut hodge = Phasor::new(0.0, 0.0);
    for e2 in 0..n {
        let c = 2.0 * psi.l1_at(e, e2);
        hodge += Phasor::new(c * psi.z[e2].re, c * psi.z[e2].im);
    }
    let mut cas = Phasor::new(0.0, 0.0);
    for &(a, b) in &psi.casimir {
        if psi.t.bridge_edges(a, b).contains(&e) {
            cas += Phasor::new(
                2.0 * cfg.casimir_kappa * psi.z[e].re,
                2.0 * cfg.casimir_kappa * psi.z[e].im,
            );
        }
    }
    hop + reg + hodge + cas
}

pub fn relax_step(psi: &mut FieldState, cfg: &FieldConfig, rng: &mut Xoshiro256StarStar) {
    let n = psi.n();
    let mut next = psi.z.clone();
    for e in 0..n {
        if psi.m[e] >= 1.0 {
            next[e] = psi.z_past[e];
            continue;
        }
        let g = energy_gradient(psi, cfg, e);
        let mix = attention_mix(psi, e);
        let noise = Phasor::new(rng.gen::<f64>() * 2.0 - 1.0, rng.gen::<f64>() * 2.0 - 1.0)
            * (2.0 * cfg.dt * cfg.temperature).sqrt();
        next[e] = psi.z[e] - g * cfg.dt + (mix - psi.z[e]) * (0.15 * cfg.dt) + noise;
    }
    psi.z = next;
}

pub fn relax(psi: &mut FieldState, cfg: &FieldConfig, steps: usize, seed: u64) {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    for _ in 0..steps {
        relax_step(psi, cfg, &mut rng);
    }
}

pub fn handshake(psi: &mut FieldState, cfg: &FieldConfig) -> f64 {
    let hs = hopfield_handshake(&psi.z_past, &psi.z);
    if hs >= cfg.handshake_threshold {
        for e in 0..psi.n() {
            let aligned = (psi.z_past[e].conj() * psi.z[e]).re
                / (psi.z_past[e].norm() * psi.z[e].norm() + EPS);
            if psi.z[e].norm() >= cfg.rigid_amp && aligned >= cfg.handshake_threshold {
                psi.m[e] = 1.0;
                psi.z_past[e] = psi.z[e];
            }
        }
    }
    hs
}

pub fn ignite_casimir(psi: &mut FieldState) {
    psi.casimir.clear();
    let n_faces = psi.t.faces.len();
    for a in 0..n_faces {
        if !face_is_rigid(psi, a) {
            continue;
        }
        for b in (a + 1)..n_faces {
            if face_is_rigid(psi, b) {
                psi.casimir.push((a, b));
            }
        }
    }
}

fn face_is_rigid(psi: &FieldState, face: usize) -> bool {
    let vs = psi.t.faces[face];
    let pairs = [(vs[0], vs[1]), (vs[1], vs[2]), (vs[0], vs[2])];
    pairs.iter().all(|&(a, b)| {
        psi.t
            .edge_index(a, b)
            .map(|e| psi.m[e] >= 1.0)
            .unwrap_or(false)
    })
}

/// Hebb simplicial local: ΔL1 ∝ −z z† en aristas del hecho que comparten frontera.
pub fn hebb_update(psi: &mut FieldState, cfg: &FieldConfig) {
    let n = psi.n();
    for e in 0..n {
        if psi.m[e] < 0.5 {
            continue;
        }
        for e2 in 0..n {
            if e2 < e || psi.m[e2] < 0.5 || !psi.t.share_boundary(e, e2) {
                continue;
            }
            let corr = (psi.z[e].conj() * psi.z[e2]).re;
            let delta = -cfg.hebb_eta * corr;
            *psi.l1_at_mut(e, e2) += delta;
            if e != e2 {
                *psi.l1_at_mut(e2, e) += delta;
            }
        }
    }
}

pub fn prune(psi: &mut FieldState, cfg: &FieldConfig) -> usize {
    let mut live = 0;
    for e in 0..psi.n() {
        if psi.m[e] >= 1.0 {
            live += 1;
            continue;
        }
        if psi.z[e].norm() < cfg.prune_eps {
            psi.z[e] = Phasor::new(0.0, 0.0);
        } else {
            live += 1;
        }
    }
    live
}

/// Patrón de campo: dos hemisferios en fase (norte 0, sur π/2).
pub fn write_hemisphere_pattern(psi: &mut FieldState) {
    for (e, _) in psi.t.edges.iter().enumerate() {
        let phase = if e < 4 {
            0.0
        } else {
            std::f64::consts::FRAC_PI_2
        };
        psi.z[e] = Phasor::from_polar(1.0, phase);
    }
    psi.z_past = psi.z.clone();
}

pub fn reconstruction_error(hidden: &[Phasor], recon: &[Phasor], mask: &[usize]) -> f64 {
    let mut acc = 0.0;
    for &e in mask {
        acc += (hidden[e] - recon[e]).norm_sqr();
    }
    acc / mask.len() as f64
}

pub fn run_cycle(seed: u64) -> CycleReport {
    let cfg = FieldConfig::default();
    let mut psi = FieldState::new(ComplexT::octahedron());
    write_hemisphere_pattern(&mut psi);
    hebb_update(&mut psi, &cfg);
    let _ = handshake(&mut psi, &cfg);
    ignite_casimir(&mut psi);

    let target = psi.z.clone();
    let mask = [0usize, 1, 8, 9];
    let baseline = reconstruction_error(&target, &vec![Phasor::new(0.0, 0.0); psi.n()], &mask);

    let energy_before = {
        for &e in &mask {
            psi.z[e] = Phasor::new(0.0, 0.0);
            psi.m[e] = 0.0;
        }
        free_energy(&psi, &cfg).total
    };

    relax(&mut psi, &cfg, 80, seed.wrapping_add(17));
    let hs = handshake(&mut psi, &cfg);
    ignite_casimir(&mut psi);
    hebb_update(&mut psi, &cfg);
    let live = prune(&mut psi, &cfg);
    let energy_after = free_energy(&psi, &cfg).total;
    let recon = reconstruction_error(&target, &psi.z, &mask);

    CycleReport {
        energy_before,
        energy_after,
        handshake: hs,
        rigid_edges: psi.m.iter().filter(|&&v| v >= 1.0).count(),
        casimir_cavities: psi.casimir.len(),
        live_edges: live,
        recon_error: recon,
        baseline_error: baseline,
    }
}

pub fn run_experiment_table(seeds: &[u64]) -> Vec<CycleReport> {
    seeds.iter().copied().map(run_cycle).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_has_no_token_vocabulary() {
        let psi = FieldState::new(ComplexT::octahedron());
        assert!(!psi.has_token_ids());
        assert_eq!(psi.n(), 12);
        assert_eq!(psi.t.faces.len(), 8);
        assert_eq!(psi.z.len(), psi.n());
    }

    #[test]
    fn geometric_attention_vanishes_without_shared_boundary() {
        let mut psi = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut psi);
        // (0,2) edge 0 and (1,4) edge 6: vértices {0,2} vs {1,4}, disjuntos.
        assert!(!psi.t.share_boundary(0, 6));
        assert_eq!(geometric_affinity(&psi, 0, 6), 0.0);
        assert!(psi.t.share_boundary(0, 1));
        assert_ne!(geometric_affinity(&psi, 0, 1), 0.0);
    }

    #[test]
    fn handshake_rigidifies_only_when_phases_lock() {
        let cfg = FieldConfig::default();
        let mut psi = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut psi);
        let hs_ok = handshake(&mut psi, &cfg);
        assert!(hs_ok >= cfg.handshake_threshold);
        assert!(psi.m.iter().filter(|&&v| v >= 1.0).count() >= 8);

        let mut noisy = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut noisy);
        for z in noisy.z.iter_mut() {
            *z = Phasor::from_polar(1.0, 2.4);
        }
        let hs_bad = handshake(&mut noisy, &cfg);
        assert!(hs_bad < cfg.handshake_threshold);
        assert!(noisy.m.iter().all(|&v| v < 1.0));
    }

    #[test]
    fn relax_does_not_raise_energy_on_attractor_perturbation() {
        let cfg = FieldConfig {
            temperature: 0.0,
            ..FieldConfig::default()
        };
        let mut psi = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut psi);
        hebb_update(&mut psi, &cfg);
        let _ = handshake(&mut psi, &cfg);
        psi.z[2] *= 0.4;
        psi.z[3] += Phasor::new(0.3, -0.2);
        psi.m[2] = 0.0;
        psi.m[3] = 0.0;
        let before = free_energy(&psi, &cfg).total;
        relax(&mut psi, &cfg, 40, 7);
        let after = free_energy(&psi, &cfg).total;
        assert!(
            after <= before + 1e-6,
            "F should not rise: before={before} after={after}"
        );
    }

    #[test]
    fn field_prediction_beats_do_nothing_baseline() {
        let report = run_cycle(0);
        assert!(
            report.recon_error < report.baseline_error,
            "recon {} should beat baseline {}",
            report.recon_error,
            report.baseline_error
        );
    }

    #[test]
    fn experiment_table_is_deterministic_and_prints() {
        let seeds: Vec<u64> = (0..8).collect();
        let rows = run_experiment_table(&seeds);
        assert_eq!(rows.len(), 8);
        let again = run_experiment_table(&seeds);
        for (a, b) in rows.iter().zip(again.iter()) {
            assert!((a.recon_error - b.recon_error).abs() < 1e-12);
        }
        println!("seed\tF_before\tF_after\tHS\trigid\tcasimir\tlive\trecon\tbaseline");
        for (seed, r) in seeds.iter().zip(rows.iter()) {
            println!(
                "{}\t{:.4}\t{:.4}\t{:.3}\t{}\t{}\t{}\t{:.4}\t{:.4}",
                seed,
                r.energy_before,
                r.energy_after,
                r.handshake,
                r.rigid_edges,
                r.casimir_cavities,
                r.live_edges,
                r.recon_error,
                r.baseline_error
            );
        }
        let mean_recon: f64 = rows.iter().map(|r| r.recon_error).sum::<f64>() / rows.len() as f64;
        let mean_base: f64 = rows.iter().map(|r| r.baseline_error).sum::<f64>() / rows.len() as f64;
        println!("mean_recon={mean_recon:.4} mean_baseline={mean_base:.4}");
        assert!(mean_recon < mean_base);
    }
}
