//! Sustrato de campo sin tokens.
//!
//! El estado no es una secuencia de IDs. Es `Ψ = (T, z, m, C)`: complejo
//! simplicial 2D, fasores en aristas, máscara de consolidación y cavidades
//! entre caras rígidas.
//!
//! `ComplexT` es el 2-esqueleto (aristas/caras). El motor
//! `simplicial_thermodynamic_engine` opera tetraedros 3D y Hodge L0–L2; no se
//! fusionan: distinta dimensión y distinto claim.
//!
//! Inferencia = relajar `z` (Langevin + atención geométrica). Aprendizaje =
//! Hebb local sobre el atractor, no next-token. La región oculta no se deja
//! en `z_past` a la hora de evaluar.

#![allow(clippy::needless_range_loop)] // índices de aristas/caras, no iteradores de colección

use crate::native_checkpoint::atomic_write;
use num_complex::Complex;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub type Phasor = Complex<f64>;

const EPS: f64 = 1.0e-12;
const PI: f64 = std::f64::consts::PI;

/// Complejo simplicial 2D orientado (vértices, aristas, caras).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplexT {
    pub n_vertices: usize,
    pub edges: Vec<(usize, usize)>,
    pub faces: Vec<[usize; 3]>,
    /// +1 polo norte, −1 polo sur, 0 ecuador / vértice interior.
    pub vertex_band: Vec<i8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Hemisphere {
    North,
    South,
}

impl ComplexT {
    /// Octaedro: 6 vértices, 12 aristas, 8 caras triangulares.
    /// Vértice 0 = norte, 1 = sur, 2..=5 = ecuador.
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
                (2, 5),
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
            vertex_band: vec![1, -1, 0, 0, 0, 0],
        }
    }

    /// Icosaedro: 12 vértices, 30 aristas, 20 caras. Polo 0 / polo 1, anillos 2..=6 y 7..=11.
    pub fn icosahedron() -> Self {
        Self {
            n_vertices: 12,
            edges: vec![
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (0, 6),
                (1, 7),
                (1, 8),
                (1, 9),
                (1, 10),
                (1, 11),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 2),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 7),
                (2, 7),
                (2, 11),
                (3, 7),
                (3, 8),
                (4, 8),
                (4, 9),
                (5, 9),
                (5, 10),
                (6, 10),
                (6, 11),
            ],
            faces: vec![
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 5],
                [0, 5, 6],
                [0, 6, 2],
                [1, 8, 7],
                [1, 9, 8],
                [1, 10, 9],
                [1, 11, 10],
                [1, 7, 11],
                [2, 7, 3],
                [3, 7, 8],
                [3, 8, 4],
                [4, 8, 9],
                [4, 9, 5],
                [5, 9, 10],
                [5, 10, 6],
                [6, 10, 11],
                [6, 11, 2],
                [2, 11, 7],
            ],
            vertex_band: vec![1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    /// Octaedro con cada cara subdividida una vez: 14 vértices, 36 aristas.
    pub fn subdivided_octahedron() -> Self {
        let mut t = Self::octahedron();
        t.subdivide_every_face();
        t
    }

    pub fn subdivide_every_face(&mut self) {
        let old_faces = std::mem::take(&mut self.faces);
        for vs in old_faces {
            let band_sum: i32 = vs
                .iter()
                .map(|&v| i32::from(self.vertex_band.get(v).copied().unwrap_or(0)))
                .sum();
            let band: i8 = if band_sum > 0 {
                1
            } else if band_sum < 0 {
                -1
            } else {
                0
            };
            let v = self.n_vertices;
            self.n_vertices += 1;
            self.vertex_band.push(band);
            self.add_edge(vs[0], v);
            self.add_edge(vs[1], v);
            self.add_edge(vs[2], v);
            self.faces.push([vs[0], vs[1], v]);
            self.faces.push([vs[1], vs[2], v]);
            self.faces.push([vs[2], vs[0], v]);
        }
    }

    pub fn is_icosahedron(&self) -> bool {
        self.n_vertices == 12 && self.edges.len() == 30
    }

    /// 0 = octaedro (12), 1 = icosaedro (30), 2 = icosa+1 (90), 3 = icosa+2 (270).
    pub fn scaled(level: usize) -> Self {
        match level {
            0 => Self::octahedron(),
            1 => Self::icosahedron(),
            n => {
                let mut t = Self::icosahedron();
                for _ in 1..n {
                    t.subdivide_every_face();
                }
                t
            }
        }
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn edge_index(&self, a: usize, b: usize) -> Option<usize> {
        let want = ordered_pair(a, b);
        self.edges
            .iter()
            .position(|&e| ordered_pair(e.0, e.1) == want)
    }

    /// Mitades iguales: 6 aristas norte y 6 sur en el octaedro semilla.
    pub fn edge_hemisphere(&self, a: usize, b: usize) -> Hemisphere {
        let sa = self.vertex_band.get(a).copied().unwrap_or(0);
        let sb = self.vertex_band.get(b).copied().unwrap_or(0);
        let sum = i16::from(sa) + i16::from(sb);
        if sum > 0 {
            return Hemisphere::North;
        }
        if sum < 0 {
            return Hemisphere::South;
        }
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        if self.is_icosahedron() {
            let north_ring = (2..=6).contains(&a) && (2..=6).contains(&b);
            let south_ring = (7..=11).contains(&a) && (7..=11).contains(&b);
            if north_ring {
                return Hemisphere::North;
            }
            if south_ring {
                return Hemisphere::South;
            }
            return match (a, b) {
                (2, 7) | (3, 8) | (4, 9) | (5, 10) | (6, 11) => Hemisphere::North,
                _ => Hemisphere::South,
            };
        }
        match (a, b) {
            (2, 3) | (4, 5) => Hemisphere::North,
            (3, 4) | (2, 5) => Hemisphere::South,
            _ if (a + b) % 2 == 0 => Hemisphere::North,
            _ => Hemisphere::South,
        }
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

    fn add_edge(&mut self, a: usize, b: usize) -> usize {
        if let Some(index) = self.edge_index(a, b) {
            return index;
        }
        self.edges.push(ordered_pair(a, b));
        self.edges.len() - 1
    }
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

type L1Snapshot<'a> = (&'a [(usize, usize)], &'a [f64]);

/// Estado de campo. Cero vocabulario: la dimensión es el número de aristas.
#[derive(Clone, Debug)]
pub struct FieldState {
    pub t: ComplexT,
    pub z: Vec<Phasor>,
    pub z_past: Vec<Phasor>,
    pub m: Vec<f64>,
    pub cavities: Vec<Cavity>,
    pub l1: Vec<f64>,
    /// Aristas del complejo semilla: no se podan.
    pub structural: Vec<bool>,
}

impl FieldState {
    /// Nombres de componentes. Si se añade vocabulario, el test de claves falla.
    pub const COMPONENTS: &'static [&'static str] = &[
        "complex_T",
        "phasors_z",
        "phasors_z_past",
        "mask_m",
        "cavities",
        "hodge_L1",
        "structural_edges",
    ];

    pub fn new(t: ComplexT) -> Self {
        let n = t.n_edges();
        let structural = vec![true; n];
        let mut psi = Self {
            z: vec![Phasor::new(0.0, 0.0); n],
            z_past: vec![Phasor::new(0.0, 0.0); n],
            m: vec![0.0; n],
            cavities: Vec::new(),
            l1: vec![0.0; n * n],
            structural,
            t,
        };
        psi.rebuild_l1(None);
        psi
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

    fn combinatorial_l1(t: &ComplexT) -> Vec<f64> {
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
        l1
    }

    fn rebuild_l1(&mut self, previous: Option<L1Snapshot<'_>>) {
        let comb = Self::combinatorial_l1(&self.t);
        let n = self.t.n_edges();
        let mut l1 = comb.clone();
        if let Some((old_edges, old_l1)) = previous {
            let old_n = old_edges.len();
            let old_comb = {
                let mut t_old = self.t.clone();
                t_old.edges = old_edges.to_vec();
                Self::combinatorial_l1(&t_old)
            };
            let mut remap = vec![None; old_n];
            for (old_i, &pair) in old_edges.iter().enumerate() {
                remap[old_i] = self.t.edge_index(pair.0, pair.1);
            }
            for i in 0..old_n {
                let Some(ni) = remap[i] else { continue };
                for j in 0..old_n {
                    let Some(nj) = remap[j] else { continue };
                    let hebb = old_l1[i * old_n + j] - old_comb[i * old_n + j];
                    l1[ni * n + nj] += hebb;
                }
            }
        }
        self.l1 = l1;
    }

    fn resize_state(&mut self, old_edges: Vec<(usize, usize)>) {
        let n = self.t.n_edges();
        let mut z = vec![Phasor::new(0.0, 0.0); n];
        let mut z_past = vec![Phasor::new(0.0, 0.0); n];
        let mut m = vec![0.0; n];
        let mut structural = vec![false; n];
        for (old_i, &(a, b)) in old_edges.iter().enumerate() {
            if let Some(ni) = self.t.edge_index(a, b) {
                if old_i < self.z.len() {
                    z[ni] = self.z[old_i];
                    z_past[ni] = self.z_past[old_i];
                    m[ni] = self.m[old_i];
                    structural[ni] = self.structural.get(old_i).copied().unwrap_or(false);
                }
            }
        }
        let old_l1 = self.l1.clone();
        self.z = z;
        self.z_past = z_past;
        self.m = m;
        self.structural = structural;
        self.rebuild_l1(Some((&old_edges, &old_l1)));
    }

    /// Subdivide una cara rígida: vértice nuevo y tres triángulos.
    pub fn subdivide_face(&mut self, face: usize) -> bool {
        if face >= self.t.faces.len() {
            return false;
        }
        let vs = self.t.faces[face];
        let band_sum: i32 = vs
            .iter()
            .map(|&v| i32::from(self.t.vertex_band.get(v).copied().unwrap_or(0)))
            .sum();
        let band: i8 = if band_sum > 0 {
            1
        } else if band_sum < 0 {
            -1
        } else {
            0
        };
        let old_edges = self.t.edges.clone();
        let v = self.t.n_vertices;
        self.t.n_vertices += 1;
        self.t.vertex_band.push(band);
        self.t.add_edge(vs[0], v);
        self.t.add_edge(vs[1], v);
        self.t.add_edge(vs[2], v);
        self.t.faces[face] = [vs[0], vs[1], v];
        self.t.faces.push([vs[1], vs[2], v]);
        self.t.faces.push([vs[2], vs[0], v]);
        self.resize_state(old_edges);
        true
    }

    pub fn grow_rigid_faces(&mut self, max_vertices: usize) -> usize {
        let rigid: Vec<usize> = (0..self.t.faces.len())
            .filter(|&f| face_is_rigid(self, f))
            .collect();
        let mut grown = 0;
        for face in rigid {
            if self.t.n_vertices >= max_vertices {
                break;
            }
            if face >= self.t.faces.len() || !face_is_rigid(self, face) {
                continue;
            }
            if self.subdivide_face(face) {
                grown += 1;
            }
        }
        grown
    }

    /// Quita aristas libres de amplitud nula que no son del complejo semilla.
    pub fn prune_geometry(&mut self, cfg: &FieldConfig) -> usize {
        let n = self.n();
        if n == 0 {
            return 0;
        }
        let mut keep = vec![true; n];
        let mut removed = 0;
        for e in 0..n {
            if self.structural.get(e).copied().unwrap_or(false) || self.m[e] >= 1.0 {
                continue;
            }
            if self.z[e].norm() < cfg.prune_eps {
                keep[e] = false;
                removed += 1;
            }
        }
        if removed == 0 {
            return 0;
        }
        let old_edges = self.t.edges.clone();
        let mut new_edges = Vec::new();
        for (e, &pair) in old_edges.iter().enumerate() {
            if keep[e] {
                new_edges.push(pair);
            }
        }
        let mut new_faces = Vec::new();
        for face in &self.t.faces {
            let e01 = edge_kept(&old_edges, &keep, face[0], face[1]);
            let e12 = edge_kept(&old_edges, &keep, face[1], face[2]);
            let e02 = edge_kept(&old_edges, &keep, face[0], face[2]);
            if e01 && e12 && e02 {
                new_faces.push(*face);
            }
        }
        self.t.edges = new_edges;
        self.t.faces = new_faces;
        self.resize_state(old_edges);
        removed
    }
}

fn edge_kept(edges: &[(usize, usize)], keep: &[bool], a: usize, b: usize) -> bool {
    let want = ordered_pair(a, b);
    edges
        .iter()
        .position(|&e| ordered_pair(e.0, e.1) == want)
        .map(|i| keep.get(i).copied().unwrap_or(false))
        .unwrap_or(false)
}

/// Cavidad entre dos caras rígidas. `lambda_min` es el modo más bajo de L1
/// restringido al puente: proxy discreto, no energía de Casimir QFT.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cavity {
    pub face_a: usize,
    pub face_b: usize,
    pub bridge: Vec<usize>,
    pub lambda_min: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FieldConfig {
    pub dt: f64,
    pub temperature: f64,
    pub lambda: f64,
    pub bridge_kappa: f64,
    pub vacuum_kappa: f64,
    pub handshake_threshold: f64,
    pub rigid_amp: f64,
    pub hebb_eta: f64,
    pub prune_eps: f64,
    pub max_vertices: usize,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            dt: 0.05,
            temperature: 0.002,
            lambda: 0.05,
            bridge_kappa: 0.15,
            vacuum_kappa: 0.05,
            handshake_threshold: 0.80,
            rigid_amp: 0.25,
            hebb_eta: 0.08,
            prune_eps: 0.02,
            max_vertices: 10,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct EnergyReport {
    pub hopfield: f64,
    pub hodge: f64,
    pub regularizer: f64,
    pub bridge_penalty: f64,
    pub cavity_vacuum: f64,
    pub total: f64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ComparisonReport {
    pub energy_before: f64,
    pub energy_after: f64,
    pub handshake: f64,
    pub rigid_edges: usize,
    pub cavities: usize,
    pub live_edges: usize,
    pub n_vertices: usize,
    pub n_edges: usize,
    pub grown_faces: usize,
    pub pruned_edges: usize,
    pub recon_field: f64,
    pub recon_do_nothing: f64,
    pub recon_neighbor: f64,
    pub recon_hopfield: f64,
    pub recon_hebb: f64,
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
    let mut bridge_penalty = 0.0;
    let mut cavity_vacuum = 0.0;
    for cav in &psi.cavities {
        cavity_vacuum += cfg.vacuum_kappa * cav.lambda_min;
        for &e in &cav.bridge {
            if e < n {
                bridge_penalty += cfg.bridge_kappa * psi.z[e].norm_sqr();
            }
        }
    }
    EnergyReport {
        hopfield,
        hodge,
        regularizer,
        bridge_penalty,
        cavity_vacuum,
        total: hopfield + hodge + regularizer + bridge_penalty + cavity_vacuum,
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
    let mut bridge = Phasor::new(0.0, 0.0);
    for cav in &psi.cavities {
        if cav.bridge.contains(&e) {
            bridge += Phasor::new(
                2.0 * cfg.bridge_kappa * psi.z[e].re,
                2.0 * cfg.bridge_kappa * psi.z[e].im,
            );
        }
    }
    hop + reg + hodge + bridge
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
        let mut nxt = psi.z[e] - g * cfg.dt + (mix - psi.z[e]) * (0.15 * cfg.dt) + noise;
        if !nxt.re.is_finite() || !nxt.im.is_finite() {
            nxt = psi.z[e];
        }
        let amp = nxt.norm();
        if amp > 2.0 {
            nxt *= 2.0 / amp;
        }
        next[e] = nxt;
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

pub fn ignite_cavities(psi: &mut FieldState) {
    psi.cavities.clear();
    let n_faces = psi.t.faces.len();
    for a in 0..n_faces {
        if !face_is_rigid(psi, a) {
            continue;
        }
        for b in (a + 1)..n_faces {
            if !face_is_rigid(psi, b) {
                continue;
            }
            let bridge = psi.t.bridge_edges(a, b);
            let lambda_min = restricted_l1_lambda_min(psi, &bridge);
            psi.cavities.push(Cavity {
                face_a: a,
                face_b: b,
                bridge,
                lambda_min,
            });
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

fn restricted_l1_lambda_min(psi: &FieldState, bridge: &[usize]) -> f64 {
    let k = bridge.len();
    if k == 0 {
        return 0.0;
    }
    if k == 1 {
        return psi.l1_at(bridge[0], bridge[0]).max(0.0);
    }
    let mut mat = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            mat[i * k + j] = psi.l1_at(bridge[i], bridge[j]);
        }
    }
    min_eig_symmetric(&mut mat, k).max(0.0)
}

fn min_eig_symmetric(mat: &mut [f64], k: usize) -> f64 {
    for _ in 0..24 {
        for p in 0..k {
            for q in (p + 1)..k {
                let app = mat[p * k + p];
                let aqq = mat[q * k + q];
                let apq = mat[p * k + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau == 0.0 {
                    1.0
                } else {
                    tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for r in 0..k {
                    let pr = mat[p * k + r];
                    let qr = mat[q * k + r];
                    mat[p * k + r] = c * pr - s * qr;
                    mat[q * k + r] = s * pr + c * qr;
                }
                for r in 0..k {
                    let rp = mat[r * k + p];
                    let rq = mat[r * k + q];
                    mat[r * k + p] = c * rp - s * rq;
                    mat[r * k + q] = s * rp + c * rq;
                }
            }
        }
    }
    (0..k).map(|i| mat[i * k + i]).fold(f64::INFINITY, f64::min)
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

pub fn prune_amplitude(psi: &mut FieldState, cfg: &FieldConfig) -> usize {
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

/// Patrón de campo: 6 aristas norte (fase 0) y 6 sur (fase π).
pub fn write_hemisphere_pattern(psi: &mut FieldState) {
    for (e, &(a, b)) in psi.t.edges.iter().enumerate() {
        let phase = match psi.t.edge_hemisphere(a, b) {
            Hemisphere::North => 0.0,
            Hemisphere::South => PI,
        };
        psi.z[e] = Phasor::from_polar(1.0, phase);
    }
}

/// Segundo patrón, corte meridiano (vértices pares del ecuador).
pub fn write_meridian_pattern(psi: &mut FieldState) {
    for (e, &(a, b)) in psi.t.edges.iter().enumerate() {
        let even = matches!(a, 2 | 4 | 6 | 8 | 10) || matches!(b, 2 | 4 | 6 | 8 | 10);
        let phase = if even { 0.0 } else { PI };
        psi.z[e] = Phasor::from_polar(1.0, phase);
    }
}

/// Tercer patrón: radios de los polos vs ecuador.
pub fn write_pole_equator_pattern(psi: &mut FieldState) {
    for (e, &(a, b)) in psi.t.edges.iter().enumerate() {
        let pole = a <= 1 || b <= 1;
        let phase = if pole { 0.0 } else { PI };
        psi.z[e] = Phasor::from_polar(1.0, phase);
    }
}

pub fn reconstruction_error(hidden: &[Phasor], recon: &[Phasor], mask: &[usize]) -> f64 {
    if mask.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for &e in mask {
        acc += (hidden[e] - recon[e]).norm_sqr();
    }
    acc / mask.len() as f64
}

pub fn comparison_mask() -> Vec<usize> {
    eval_mask(12)
}

pub fn eval_mask(n: usize) -> Vec<usize> {
    if n == 12 {
        return vec![0, 3, 4, 7];
    }
    let count = (n / 6).clamp(4, n.max(1) / 2);
    spaced_mask(n, count)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FieldGeometry {
    Octahedron,
    Icosahedron,
    SubdividedOctahedron,
}

impl FieldGeometry {
    pub fn complex(self) -> ComplexT {
        match self {
            Self::Octahedron => ComplexT::octahedron(),
            Self::Icosahedron => ComplexT::icosahedron(),
            Self::SubdividedOctahedron => ComplexT::subdivided_octahedron(),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Octahedron => "octahedron",
            Self::Icosahedron => "icosahedron",
            Self::SubdividedOctahedron => "subdivided-octahedron",
        }
    }

    pub fn all() -> [Self; 3] {
        [
            Self::Octahedron,
            Self::Icosahedron,
            Self::SubdividedOctahedron,
        ]
    }
}

pub fn spaced_mask(n: usize, count: usize) -> Vec<usize> {
    if n == 0 || count == 0 {
        return Vec::new();
    }
    let k = count.min(n);
    (0..k).map(|i| i * n / k).collect()
}

pub fn mask_hidden(psi: &mut FieldState, mask: &[usize]) {
    for &e in mask {
        if e >= psi.n() {
            continue;
        }
        psi.z[e] = Phasor::new(0.0, 0.0);
        psi.z_past[e] = Phasor::new(0.0, 0.0);
        psi.m[e] = 0.0;
    }
}

pub fn neighbor_mean_retrieve(psi: &FieldState, cue: &[Phasor], mask: &[usize]) -> Vec<Phasor> {
    let n = psi.n();
    let mut hidden = vec![false; n];
    for &e in mask {
        if e < n {
            hidden[e] = true;
        }
    }
    let mut z = cue.to_vec();
    if z.len() != n {
        z.resize(n, Phasor::new(0.0, 0.0));
    }
    for e in 0..n {
        if !hidden[e] {
            continue;
        }
        let mut acc = Phasor::new(0.0, 0.0);
        let mut wsum = 0.0;
        for e2 in 0..n {
            if hidden[e2] || !psi.t.share_boundary(e, e2) {
                continue;
            }
            acc += z[e2];
            wsum += 1.0;
        }
        z[e] = if wsum > EPS {
            acc / wsum
        } else {
            Phasor::new(0.0, 0.0)
        };
    }
    z
}

pub fn hebb_retrieve(
    psi: &FieldState,
    cue: &[Phasor],
    mask: &[usize],
    steps: usize,
) -> Vec<Phasor> {
    let n = psi.n();
    let mut hidden = vec![false; n];
    for &e in mask {
        if e < n {
            hidden[e] = true;
        }
    }
    let mut z = cue.to_vec();
    if z.len() != n {
        z.resize(n, Phasor::new(0.0, 0.0));
    }
    for _ in 0..steps {
        let mut next = z.clone();
        for e in 0..n {
            if !hidden[e] {
                continue;
            }
            let diag = psi.l1_at(e, e);
            if diag.abs() < EPS {
                continue;
            }
            let mut acc = Phasor::new(0.0, 0.0);
            for e2 in 0..n {
                if e2 == e {
                    continue;
                }
                acc += z[e2] * psi.l1_at(e, e2);
            }
            next[e] = acc * (-1.0 / diag);
        }
        z = next;
    }
    z
}

pub fn hopfield_retrieve(patterns: &[Vec<Phasor>], cue: &[Phasor], steps: usize) -> Vec<Phasor> {
    if patterns.is_empty() {
        return cue.to_vec();
    }
    let n = cue.len();
    let mut z = cue.to_vec();
    for _ in 0..steps {
        let mut next = vec![Phasor::new(0.0, 0.0); n];
        for mu in patterns {
            if mu.len() != n {
                continue;
            }
            let mut overlap = Phasor::new(0.0, 0.0);
            for j in 0..n {
                overlap += mu[j].conj() * z[j];
            }
            for i in 0..n {
                next[i] += mu[i] * overlap;
            }
        }
        for i in 0..n {
            let norm = next[i].norm();
            next[i] = if norm > EPS { next[i] / norm } else { z[i] };
        }
        z = next;
    }
    z
}

pub fn commit_pattern(psi: &mut FieldState, cfg: &FieldConfig) {
    psi.z_past = psi.z.clone();
    let _ = handshake(psi, cfg);
    hebb_update(psi, cfg);
    ignite_cavities(psi);
}

fn score_methods(
    trained: &FieldState,
    cfg: &FieldConfig,
    target: &[Phasor],
    patterns: &[Vec<Phasor>],
    mask: &[usize],
    seed: u64,
) -> ComparisonReport {
    let baseline = reconstruction_error(target, &vec![Phasor::new(0.0, 0.0); trained.n()], mask);

    let mut field = trained.clone();
    mask_hidden(&mut field, mask);
    ignite_cavities(&mut field);
    let energy_before = free_energy(&field, cfg).total;
    relax(&mut field, cfg, 80, seed.wrapping_add(17));
    let hs = handshake(&mut field, cfg);
    ignite_cavities(&mut field);
    hebb_update(&mut field, cfg);
    let live = prune_amplitude(&mut field, cfg);
    let energy_after = free_energy(&field, cfg).total;
    let recon_field = reconstruction_error(target, &field.z, mask);

    let mut cue_state = trained.clone();
    mask_hidden(&mut cue_state, mask);
    let neighbor = neighbor_mean_retrieve(&cue_state, &cue_state.z, mask);
    let hebb = hebb_retrieve(&cue_state, &cue_state.z, mask, 8);
    let hopfield = hopfield_retrieve(patterns, &cue_state.z, 8);

    ComparisonReport {
        energy_before,
        energy_after,
        handshake: hs,
        rigid_edges: field.m.iter().filter(|&&v| v >= 1.0).count(),
        cavities: field.cavities.len(),
        live_edges: live,
        n_vertices: field.t.n_vertices,
        n_edges: field.n(),
        grown_faces: 0,
        pruned_edges: 0,
        recon_field,
        recon_do_nothing: baseline,
        recon_neighbor: reconstruction_error(target, &neighbor, mask),
        recon_hopfield: reconstruction_error(target, &hopfield, mask),
        recon_hebb: reconstruction_error(target, &hebb, mask),
    }
}

pub fn run_comparison(seed: u64) -> ComparisonReport {
    let cfg = FieldConfig::default();
    let mut psi = FieldState::new(ComplexT::octahedron());
    write_hemisphere_pattern(&mut psi);
    commit_pattern(&mut psi, &cfg);
    let target = psi.z.clone();
    let mask = comparison_mask();
    score_methods(
        &psi,
        &cfg,
        &target,
        std::slice::from_ref(&target),
        &mask,
        seed,
    )
}

pub fn run_two_pattern_comparison(seed: u64) -> (ComparisonReport, ComparisonReport) {
    let cfg = FieldConfig::default();
    let mut psi = FieldState::new(ComplexT::octahedron());
    write_hemisphere_pattern(&mut psi);
    commit_pattern(&mut psi, &cfg);
    let first = psi.z.clone();
    write_meridian_pattern(&mut psi);
    commit_pattern(&mut psi, &cfg);
    let second = psi.z.clone();
    let patterns = vec![first.clone(), second.clone()];
    let mask = comparison_mask();

    let mut cue_a = psi.clone();
    cue_a.z = first.clone();
    cue_a.z_past = first.clone();
    let report_a = score_methods(&cue_a, &cfg, &first, &patterns, &mask, seed);

    let mut cue_b = psi.clone();
    cue_b.z = second.clone();
    cue_b.z_past = second.clone();
    let report_b = score_methods(
        &cue_b,
        &cfg,
        &second,
        &patterns,
        &mask,
        seed.wrapping_add(1),
    );
    (report_a, report_b)
}

pub fn run_experiment_table(seeds: &[u64]) -> Vec<ComparisonReport> {
    seeds.iter().copied().map(run_comparison).collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldCheckpoint {
    pub version: u32,
    pub dataset_id: String,
    pub seed: u64,
    pub patterns_total: usize,
    pub patterns_seen: usize,
    pub n_vertices: usize,
    pub vertex_band: Vec<i8>,
    pub edges: Vec<(usize, usize)>,
    pub faces: Vec<[usize; 3]>,
    pub z: Vec<[f64; 2]>,
    pub z_past: Vec<[f64; 2]>,
    pub m: Vec<f64>,
    pub structural: Vec<bool>,
    pub l1: Vec<f64>,
    pub mean_recon_field: f64,
    pub mean_recon_do_nothing: f64,
    pub mean_recon_neighbor: f64,
    pub mean_recon_hopfield: f64,
    pub mean_recon_hebb: f64,
    pub n_edges: usize,
    pub grown_faces: usize,
    pub pruned_edges: usize,
}

impl FieldCheckpoint {
    pub const VERSION: u32 = 2;
    pub const KEYS: &'static [&'static str] = &[
        "version",
        "dataset_id",
        "seed",
        "patterns_total",
        "patterns_seen",
        "n_vertices",
        "vertex_band",
        "edges",
        "faces",
        "z",
        "z_past",
        "m",
        "structural",
        "l1",
        "mean_recon_field",
        "mean_recon_do_nothing",
        "mean_recon_neighbor",
        "mean_recon_hopfield",
        "mean_recon_hebb",
        "n_edges",
        "grown_faces",
        "pruned_edges",
    ];
}

#[derive(Clone, Debug)]
pub struct DatasetTrainConfig {
    pub dataset_id: String,
    pub seed: u64,
    pub patterns_total: usize,
    pub checkpoint_dir: PathBuf,
    pub checkpoint_every: usize,
    pub resume: bool,
    pub mask_count: usize,
    pub relax_steps: usize,
    pub grow: bool,
    pub prune_geometry: bool,
    pub max_vertices: usize,
    pub stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl DatasetTrainConfig {
    pub fn synthetic(dir: impl Into<PathBuf>, patterns_total: usize) -> Self {
        Self {
            dataset_id: "synthetic-unit-phasors".to_string(),
            seed: 0xF1E1_D501,
            patterns_total,
            checkpoint_dir: dir.into(),
            checkpoint_every: 8.max(patterns_total / 8).max(1),
            resume: true,
            mask_count: 4,
            relax_steps: 40,
            grow: true,
            prune_geometry: true,
            max_vertices: 10,
            stop: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingProgress {
    pub dataset_id: String,
    pub patterns_seen: usize,
    pub patterns_total: usize,
    pub mean_recon_field: f64,
    pub mean_recon_do_nothing: f64,
    pub mean_recon_neighbor: f64,
    pub mean_recon_hopfield: f64,
    pub mean_recon_hebb: f64,
    pub n_vertices: usize,
    pub n_edges: usize,
    pub grown_faces: usize,
    pub pruned_edges: usize,
    pub checkpoint_path: String,
    pub metrics_path: String,
    pub resumed_from: usize,
    pub interrupted: bool,
}

pub fn synthetic_pattern(n_edges: usize, seed: u64, index: usize) -> Vec<Phasor> {
    let mix = seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rng = Xoshiro256StarStar::seed_from_u64(mix);
    (0..n_edges)
        .map(|_| Phasor::from_polar(1.0, rng.gen::<f64>() * 2.0 * PI))
        .collect()
}

fn phasors_to_pairs(z: &[Phasor]) -> Vec<[f64; 2]> {
    z.iter().map(|p| [p.re, p.im]).collect()
}

fn pairs_to_phasors(pairs: &[[f64; 2]]) -> Vec<Phasor> {
    pairs.iter().map(|p| Phasor::new(p[0], p[1])).collect()
}

fn state_from_checkpoint(ckpt: &FieldCheckpoint) -> FieldState {
    FieldState {
        t: ComplexT {
            n_vertices: ckpt.n_vertices,
            edges: ckpt.edges.clone(),
            faces: ckpt.faces.clone(),
            vertex_band: ckpt.vertex_band.clone(),
        },
        z: pairs_to_phasors(&ckpt.z),
        z_past: pairs_to_phasors(&ckpt.z_past),
        m: ckpt.m.clone(),
        cavities: Vec::new(),
        l1: ckpt.l1.clone(),
        structural: ckpt.structural.clone(),
    }
}

fn checkpoint_from_state(
    cfg: &DatasetTrainConfig,
    psi: &FieldState,
    seen: usize,
    means: &[f64; 5],
    grown: usize,
    pruned: usize,
) -> FieldCheckpoint {
    FieldCheckpoint {
        version: FieldCheckpoint::VERSION,
        dataset_id: cfg.dataset_id.clone(),
        seed: cfg.seed,
        patterns_total: cfg.patterns_total,
        patterns_seen: seen,
        n_vertices: psi.t.n_vertices,
        vertex_band: psi.t.vertex_band.clone(),
        edges: psi.t.edges.clone(),
        faces: psi.t.faces.clone(),
        z: phasors_to_pairs(&psi.z),
        z_past: phasors_to_pairs(&psi.z_past),
        m: psi.m.clone(),
        structural: psi.structural.clone(),
        l1: psi.l1.clone(),
        mean_recon_field: means[0],
        mean_recon_do_nothing: means[1],
        mean_recon_neighbor: means[2],
        mean_recon_hopfield: means[3],
        mean_recon_hebb: means[4],
        n_edges: psi.n(),
        grown_faces: grown,
        pruned_edges: pruned,
    }
}

pub fn save_training_checkpoint(path: &Path, ckpt: &FieldCheckpoint) -> Result<(), String> {
    let body = serde_json::to_vec_pretty(ckpt).map_err(|e| e.to_string())?;
    atomic_write(path, &body)
}

pub fn load_training_checkpoint(path: &Path) -> Result<FieldCheckpoint, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn append_metrics(path: &Path, line: &serde_json::Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    writeln!(file, "{line}").map_err(|e| e.to_string())?;
    file.flush().map_err(|e| e.to_string())
}

fn apply_pattern(psi: &mut FieldState, pattern: &[Phasor]) {
    let n = psi.n().min(pattern.len());
    psi.z[..n].copy_from_slice(&pattern[..n]);
}

/// Entrena un dataset de patrones sintéticos, con resume atómico por `dataset_id`.
pub fn run_dataset_training(cfg: DatasetTrainConfig) -> Result<TrainingProgress, String> {
    let root = cfg.checkpoint_dir.clone();
    fs::create_dir_all(root.join("checkpoints")).map_err(|e| e.to_string())?;
    let latest = root.join("latest.json");
    let metrics_path = root.join("metrics.jsonl");
    let field_cfg = FieldConfig {
        max_vertices: cfg.max_vertices,
        ..FieldConfig::default()
    };

    let mut psi = FieldState::new(ComplexT::octahedron());
    let mut seen = 0usize;
    let mut grown_total = 0usize;
    let mut pruned_total = 0usize;
    let mut sum = [0.0f64; 5];
    let mut resumed_from = 0usize;

    if cfg.resume && latest.exists() {
        let ckpt = load_training_checkpoint(&latest)?;
        if ckpt.dataset_id != cfg.dataset_id || ckpt.seed != cfg.seed {
            return Err(format!(
                "checkpoint de dataset '{}' seed {} no coincide con '{}' seed {}",
                ckpt.dataset_id, ckpt.seed, cfg.dataset_id, cfg.seed
            ));
        }
        if ckpt.version != FieldCheckpoint::VERSION {
            return Err(format!(
                "checkpoint version {} != {}",
                ckpt.version,
                FieldCheckpoint::VERSION
            ));
        }
        psi = state_from_checkpoint(&ckpt);
        seen = ckpt.patterns_seen.min(cfg.patterns_total);
        resumed_from = seen;
        grown_total = ckpt.grown_faces;
        pruned_total = ckpt.pruned_edges;
        if seen > 0 {
            sum[0] = ckpt.mean_recon_field * seen as f64;
            sum[1] = ckpt.mean_recon_do_nothing * seen as f64;
            sum[2] = ckpt.mean_recon_neighbor * seen as f64;
            sum[3] = ckpt.mean_recon_hopfield * seen as f64;
            sum[4] = ckpt.mean_recon_hebb * seen as f64;
        }
    } else if !cfg.resume && metrics_path.exists() {
        let _ = fs::remove_file(&metrics_path);
    }

    let mut interrupted = false;
    let mut hopfield_memory: Vec<Vec<Phasor>> = Vec::new();
    let memory_cap = 16usize;

    while seen < cfg.patterns_total {
        if cfg
            .stop
            .as_ref()
            .is_some_and(|s| s.load(std::sync::atomic::Ordering::SeqCst))
        {
            interrupted = true;
            break;
        }
        let pattern = synthetic_pattern(psi.n(), cfg.seed, seen);
        apply_pattern(&mut psi, &pattern);
        commit_pattern(&mut psi, &field_cfg);
        if hopfield_memory.len() >= memory_cap {
            hopfield_memory.remove(0);
        }
        hopfield_memory.push(pattern.clone());

        let mask = spaced_mask(psi.n(), cfg.mask_count);
        let mut eval = psi.clone();
        mask_hidden(&mut eval, &mask);
        ignite_cavities(&mut eval);
        relax(
            &mut eval,
            &field_cfg,
            cfg.relax_steps,
            cfg.seed.wrapping_add(seen as u64),
        );
        let recon_field = reconstruction_error(&pattern, &eval.z, &mask);
        let recon_zero =
            reconstruction_error(&pattern, &vec![Phasor::new(0.0, 0.0); psi.n()], &mask);
        let recon_neighbor = reconstruction_error(
            &pattern,
            &neighbor_mean_retrieve(&eval, &eval.z, &mask),
            &mask,
        );
        let recon_hebb =
            reconstruction_error(&pattern, &hebb_retrieve(&eval, &eval.z, &mask, 8), &mask);
        let recon_hop = reconstruction_error(
            &pattern,
            &hopfield_retrieve(&hopfield_memory, &eval.z, 8),
            &mask,
        );
        sum[0] += recon_field;
        sum[1] += recon_zero;
        sum[2] += recon_neighbor;
        sum[3] += recon_hop;
        sum[4] += recon_hebb;

        if cfg.prune_geometry {
            pruned_total += psi.prune_geometry(&field_cfg);
        }
        if cfg.grow {
            grown_total += psi.grow_rigid_faces(cfg.max_vertices);
        }

        seen += 1;
        let denom = seen as f64;
        let means = [
            sum[0] / denom,
            sum[1] / denom,
            sum[2] / denom,
            sum[3] / denom,
            sum[4] / denom,
        ];
        let line = serde_json::json!({
            "dataset_id": cfg.dataset_id,
            "pattern_index": seen - 1,
            "n_vertices": psi.t.n_vertices,
            "n_edges": psi.n(),
            "recon_field": recon_field,
            "recon_do_nothing": recon_zero,
            "recon_neighbor": recon_neighbor,
            "recon_hopfield": recon_hop,
            "recon_hebb": recon_hebb,
        });
        append_metrics(&metrics_path, &line)?;

        let should_ckpt = seen.is_multiple_of(cfg.checkpoint_every) || seen == cfg.patterns_total;
        if should_ckpt {
            persist_progress(&cfg, &latest, &psi, seen, &means, grown_total, pruned_total)?;
        }
    }

    let denom = seen.max(1) as f64;
    let means = [
        sum[0] / denom,
        sum[1] / denom,
        sum[2] / denom,
        sum[3] / denom,
        sum[4] / denom,
    ];
    persist_progress(&cfg, &latest, &psi, seen, &means, grown_total, pruned_total)?;
    let summary = TrainingProgress {
        dataset_id: cfg.dataset_id.clone(),
        patterns_seen: seen,
        patterns_total: cfg.patterns_total,
        mean_recon_field: means[0],
        mean_recon_do_nothing: means[1],
        mean_recon_neighbor: means[2],
        mean_recon_hopfield: means[3],
        mean_recon_hebb: means[4],
        n_vertices: psi.t.n_vertices,
        n_edges: psi.n(),
        grown_faces: grown_total,
        pruned_edges: pruned_total,
        checkpoint_path: latest.display().to_string(),
        metrics_path: metrics_path.display().to_string(),
        resumed_from,
        interrupted,
    };
    let summary_path = root.join("summary.json");
    atomic_write(
        &summary_path,
        serde_json::to_vec_pretty(&summary)
            .map_err(|e| e.to_string())?
            .as_slice(),
    )?;
    Ok(summary)
}

fn persist_progress(
    cfg: &DatasetTrainConfig,
    latest: &Path,
    psi: &FieldState,
    seen: usize,
    means: &[f64; 5],
    grown: usize,
    pruned: usize,
) -> Result<(), String> {
    let ckpt = checkpoint_from_state(cfg, psi, seen, means, grown, pruned);
    save_training_checkpoint(latest, &ckpt)?;
    let step = cfg
        .checkpoint_dir
        .join("checkpoints")
        .join(format!("step-{:09}.json", seen));
    save_training_checkpoint(&step, &ckpt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_components_are_not_a_token_vocabulary() {
        for name in FieldState::COMPONENTS {
            let lower = name.to_ascii_lowercase();
            assert!(
                !lower.contains("token") && !lower.contains("vocab") && !lower.contains("id"),
                "component {name} looks like a vocabulary slot"
            );
        }
        for key in FieldCheckpoint::KEYS {
            let lower = key.to_ascii_lowercase();
            assert!(
                !lower.contains("token") && !lower.contains("vocab"),
                "checkpoint key {key} looks like a vocabulary slot"
            );
        }
        let psi = FieldState::new(ComplexT::octahedron());
        assert_eq!(psi.n(), 12);
        assert_eq!(psi.z.len(), psi.t.n_edges());
        assert_eq!(psi.t.faces.len(), 8);
        let names = format!("{:?}", FieldState::COMPONENTS);
        assert!(!names.contains("token_ids"));
    }

    #[test]
    fn hemisphere_pattern_splits_equal_north_south() {
        let t = ComplexT::octahedron();
        let mut north = 0;
        let mut south = 0;
        for &(a, b) in &t.edges {
            match t.edge_hemisphere(a, b) {
                Hemisphere::North => north += 1,
                Hemisphere::South => south += 1,
            }
        }
        assert_eq!(north, 6, "north={north} south={south}");
        assert_eq!(south, 6, "north={north} south={south}");
        let mut psi = FieldState::new(t);
        write_hemisphere_pattern(&mut psi);
        let n0 = psi
            .z
            .iter()
            .filter(|p| (p.re - 1.0).abs() < 1e-9 && p.im.abs() < 1e-9)
            .count();
        let npi = psi
            .z
            .iter()
            .filter(|p| (p.re + 1.0).abs() < 1e-9 && p.im.abs() < 1e-9)
            .count();
        assert_eq!(n0, 6);
        assert_eq!(npi, 6);
    }

    #[test]
    fn icosahedron_has_thirty_balanced_edges() {
        let t = ComplexT::icosahedron();
        assert_eq!(t.n_vertices, 12);
        assert_eq!(t.n_edges(), 30);
        assert_eq!(t.faces.len(), 20);
        let mut north = 0;
        let mut south = 0;
        for &(a, b) in &t.edges {
            match t.edge_hemisphere(a, b) {
                Hemisphere::North => north += 1,
                Hemisphere::South => south += 1,
            }
        }
        assert_eq!(north, 15, "north={north} south={south}");
        assert_eq!(south, 15, "north={north} south={south}");
    }

    #[test]
    fn relax_stays_bounded_on_wide_substrate() {
        let cfg = FieldConfig {
            temperature: 0.0,
            ..FieldConfig::default()
        };
        let mut psi = FieldState::new(ComplexT::scaled(2));
        write_hemisphere_pattern(&mut psi);
        commit_pattern(&mut psi, &cfg);
        let mask = eval_mask(psi.n());
        mask_hidden(&mut psi, &mask);
        relax(&mut psi, &cfg, 40, 9);
        assert!(psi.z.iter().all(|p| p.re.is_finite() && p.im.is_finite()));
        assert!(psi.z.iter().all(|p| p.norm() <= 2.0 + 1e-9));
        let recon = reconstruction_error(
            &vec![Phasor::from_polar(1.0, 0.0); psi.n()],
            &psi.z,
            &mask,
        );
        assert!(recon.is_finite());
        assert!(recon < 100.0, "recon exploded: {recon}");
    }

    #[test]
    fn scaled_complex_widens_the_substrate() {
        assert_eq!(ComplexT::scaled(0).n_edges(), 12);
        assert_eq!(ComplexT::scaled(1).n_edges(), 30);
        let t2 = ComplexT::scaled(2);
        assert_eq!(t2.n_edges(), 90);
        assert!(t2.n_vertices > 12);
        let t3 = ComplexT::scaled(3);
        assert_eq!(t3.n_edges(), 270);
        let sub = ComplexT::subdivided_octahedron();
        assert_eq!(sub.n_edges(), 36);
        assert_eq!(sub.n_vertices, 14);
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
        psi.z_past = psi.z.clone();
        let hs_ok = handshake(&mut psi, &cfg);
        assert!(hs_ok >= cfg.handshake_threshold);
        assert!(psi.m.iter().filter(|&&v| v >= 1.0).count() >= 8);

        let mut noisy = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut noisy);
        noisy.z_past = noisy.z.clone();
        for z in noisy.z.iter_mut() {
            *z = Phasor::from_polar(1.0, 2.4);
        }
        let hs_bad = handshake(&mut noisy, &cfg);
        assert!(hs_bad < cfg.handshake_threshold);
        assert!(noisy.m.iter().all(|&v| v < 1.0));
    }

    #[test]
    fn hidden_region_is_not_stored_in_z_past() {
        let cfg = FieldConfig::default();
        let mut psi = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut psi);
        commit_pattern(&mut psi, &cfg);
        let mask = comparison_mask();
        mask_hidden(&mut psi, &mask);
        for &e in &mask {
            assert!(psi.z[e].norm() < EPS);
            assert!(psi.z_past[e].norm() < EPS);
            assert!(psi.m[e] < 1.0);
        }
    }

    #[test]
    fn relax_does_not_raise_energy_on_attractor_perturbation() {
        let cfg = FieldConfig {
            temperature: 0.0,
            ..FieldConfig::default()
        };
        let mut psi = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut psi);
        commit_pattern(&mut psi, &cfg);
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
    fn geometry_grows_and_prunes_off_the_seed() {
        let cfg = FieldConfig::default();
        let mut psi = FieldState::new(ComplexT::octahedron());
        write_hemisphere_pattern(&mut psi);
        commit_pattern(&mut psi, &cfg);
        let v0 = psi.t.n_vertices;
        let e0 = psi.n();
        let grown = psi.grow_rigid_faces(8);
        assert!(grown >= 1);
        assert!(psi.t.n_vertices > v0);
        assert!(psi.n() > e0);
        assert!(psi.structural.iter().filter(|&&s| s).count() == 12);
        psi.z[e0] = Phasor::new(0.0, 0.0);
        psi.m[e0] = 0.0;
        let pruned = psi.prune_geometry(&cfg);
        assert!(pruned >= 1);
        assert!(psi.n() < e0 + 3 * grown);
    }

    #[test]
    fn field_and_baselines_beat_do_nothing() {
        let report = run_comparison(0);
        assert!(
            report.recon_field < report.recon_do_nothing,
            "field {} should beat zeros {}",
            report.recon_field,
            report.recon_do_nothing
        );
        assert!(
            report.recon_neighbor < report.recon_do_nothing,
            "neighbor {} should beat zeros {}",
            report.recon_neighbor,
            report.recon_do_nothing
        );
        assert!(
            report.recon_hopfield < report.recon_do_nothing,
            "hopfield {} should beat zeros {}",
            report.recon_hopfield,
            report.recon_do_nothing
        );
        assert!(
            report.recon_hebb < report.recon_do_nothing,
            "hebb {} should beat zeros {}",
            report.recon_hebb,
            report.recon_do_nothing
        );
        assert!((report.recon_do_nothing - 1.0).abs() < 1e-9);
    }

    #[test]
    fn two_patterns_are_scored_without_leaking_the_hidden_region() {
        let (a, b) = run_two_pattern_comparison(0);
        assert!(a.recon_field < a.recon_do_nothing);
        assert!(b.recon_hopfield < b.recon_do_nothing);
        assert!(a.recon_hopfield.is_finite());
        assert!(b.recon_hebb.is_finite());
    }

    #[test]
    fn experiment_table_is_deterministic_and_prints() {
        let seeds: Vec<u64> = (0..8).collect();
        let rows = run_experiment_table(&seeds);
        assert_eq!(rows.len(), 8);
        let again = run_experiment_table(&seeds);
        for (a, b) in rows.iter().zip(again.iter()) {
            assert!((a.recon_field - b.recon_field).abs() < 1e-12);
            assert!((a.recon_hopfield - b.recon_hopfield).abs() < 1e-12);
        }
        println!(
            "seed\tF_before\tF_after\tHS\trigid\tcav\tverts\tedges\tfield\tzero\tneigh\thop\thebb"
        );
        for (seed, r) in seeds.iter().zip(rows.iter()) {
            println!(
                "{}\t{:.4}\t{:.4}\t{:.3}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                seed,
                r.energy_before,
                r.energy_after,
                r.handshake,
                r.rigid_edges,
                r.cavities,
                r.n_vertices,
                r.n_edges,
                r.recon_field,
                r.recon_do_nothing,
                r.recon_neighbor,
                r.recon_hopfield,
                r.recon_hebb
            );
        }
        let mean =
            |f: fn(&ComparisonReport) -> f64| rows.iter().map(f).sum::<f64>() / rows.len() as f64;
        let mean_field = mean(|r| r.recon_field);
        let mean_zero = mean(|r| r.recon_do_nothing);
        let mean_neigh = mean(|r| r.recon_neighbor);
        let mean_hop = mean(|r| r.recon_hopfield);
        let mean_hebb = mean(|r| r.recon_hebb);
        println!(
            "mean_field={mean_field:.4} mean_zero={mean_zero:.4} mean_neigh={mean_neigh:.4} mean_hop={mean_hop:.4} mean_hebb={mean_hebb:.4}"
        );
        assert!(mean_field < mean_zero);
        assert!(mean_neigh < mean_zero);
        assert!(mean_hop < mean_zero);
        assert!(mean_hebb < mean_zero);
    }

    #[test]
    fn dataset_checkpoint_resumes_the_same_dataset() {
        let dir = std::env::temp_dir().join(format!(
            "field_sub_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&dir);
        let mut cfg = DatasetTrainConfig::synthetic(&dir, 8);
        cfg.checkpoint_every = 3;
        cfg.grow = true;
        cfg.prune_geometry = true;
        cfg.resume = false;
        let first = run_dataset_training(cfg.clone()).expect("train 8");
        assert_eq!(first.patterns_seen, 8);
        assert_eq!(first.resumed_from, 0);
        assert!(Path::new(&first.checkpoint_path).exists());
        assert!(Path::new(&first.metrics_path).exists());
        assert!(first.n_vertices >= 6);

        cfg.patterns_total = 12;
        cfg.resume = true;
        let resumed = run_dataset_training(cfg).expect("resume 8->12");
        assert_eq!(resumed.resumed_from, 8);
        assert_eq!(resumed.patterns_seen, 12);
        assert_eq!(resumed.dataset_id, first.dataset_id);

        let ckpt = load_training_checkpoint(Path::new(&resumed.checkpoint_path)).unwrap();
        let value = serde_json::to_value(&ckpt).unwrap();
        let map = value.as_object().unwrap();
        for key in map.keys() {
            let lower = key.to_ascii_lowercase();
            assert!(!lower.contains("token") && !lower.contains("vocab"));
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
