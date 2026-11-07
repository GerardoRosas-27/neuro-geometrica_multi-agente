//! Soft-Bellman Simplicial 3D v3 GPU — Complejos Tetraédricos Orientados con:
//! 1. Aceleración GPU nativa mediante WGPU Compute Shaders (AMD Radeon RX 7800 XT).
//! 2. Optimización Continua Baricéntrica de Snell-Fermat (\lambda^* \in \Delta^k) y refinamiento multinivel.
//! 3. Base No Lineal de Fourier-Polinomios + Adaptación / Gate Epistémico ante Cambio de Ley del Terreno.
//! 4. Operadores Entrenados de Hodge (\Delta_1), Curvatura Intrínseca de Regge (S_Regge), Barrera de
//!    Cayley-Menger (B_CM) y Holonomía de 2-Cocadena Magnética (B = d_1 A con d_2 B = 0).
//! 5. Importación y Exportación de Mallas Tetraédricas 3D Externas (.mesh Medit, .vtk VTK, .json).

pub mod continuous_geodesic;
pub mod external_mesh;
pub mod gpu_simplicial;

use continuous_geodesic::optimize_barycentric_snell_polyline;
use external_mesh::{
    build_external_l_corridor_mesh, build_external_snell_prism_mesh, ExternalSimplicialFile,
};
use gpu_simplicial::{GpuEdgeInput, GpuExecutionTelemetry, GpuSimplicialEngine, GpuStateSpan, GpuTetInput};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

const P: usize = 9;
pub const P_V3: usize = 18;
const TAU: f64 = 0.03;
const EPS: f64 = 0.01;
const EPOCHS: usize = 120;
type V = [f64; 3];

#[derive(Clone)]
struct Rng {
    s: u64,
    spare: Option<f64>,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            s: seed,
            spare: None,
        }
    }
    fn next(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / 9007199254740992.0
    }
    fn usize(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
    fn normal(&mut self) -> f64 {
        if let Some(x) = self.spare.take() {
            return x;
        }
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let a = std::f64::consts::TAU * self.uniform();
        self.spare = Some(r * a.sin());
        r * a.cos()
    }
}

fn sp(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}
fn sig(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}
fn dot(w: &[f64], f: &[f64]) -> f64 {
    w.iter().zip(f).map(|(a, b)| a * b).sum()
}
fn softmin(q: &[f64]) -> (f64, Vec<f64>) {
    let m = q.iter().copied().fold(f64::INFINITY, f64::min);
    if !m.is_finite() {
        return (f64::INFINITY, vec![0.0; q.len()]);
    }
    let mut p: Vec<f64> = q.iter().map(|x| (-(x - m) / TAU).exp()).collect();
    let z: f64 = p.iter().sum();
    for v in &mut p {
        *v /= z;
    }
    (m - TAU * z.ln(), p)
}

fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V, k: f64) -> V {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot3(a: V, b: V) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn det3(a: V, b: V, c: V) -> f64 {
    dot3(a, cross(b, c))
}
fn transform(d: V, w: &[f64]) -> V {
    [
        (0.5 + sp(w[0])) * d[0] + 0.8 * w[3].tanh() * d[1] + 0.8 * w[4].tanh() * d[2],
        (0.5 + sp(w[1])) * d[1] + 0.8 * w[5].tanh() * d[2],
        (0.5 + sp(w[2])) * d[2],
    ]
}
fn inner(a: V, b: V, w: &[f64]) -> f64 {
    dot3(transform(a, w), transform(b, w)) + EPS * dot3(a, b)
}
fn length(d: V, w: &[f64]) -> (f64, [f64; P]) {
    let v = transform(d, w);
    let n = (dot3(v, v) + EPS * dot3(d, d)).sqrt();
    assert!(n > 1e-10);
    let mut g = [0.0; P];
    for j in 0..3 {
        g[j] = v[j] * d[j] * sig(w[j]) / n;
    }
    g[3] = v[0] * d[1] * 0.8 * (1.0 - w[3].tanh().powi(2)) / n;
    g[4] = v[0] * d[2] * 0.8 * (1.0 - w[4].tanh().powi(2)) / n;
    g[5] = v[1] * d[2] * 0.8 * (1.0 - w[5].tanh().powi(2)) / n;
    (n, g)
}

/// Gradiente espacial \partial \|d\|_{G_w} / \partial d \in \mathbb{R}^3 para optimización baricéntrica continua
fn length_and_spatial_grad(d: V, w: &[f64]) -> (f64, V) {
    let l00 = 0.5 + sp(w[0]);
    let l11 = 0.5 + sp(w[1]);
    let l22 = 0.5 + sp(w[2]);
    let l01 = 0.8 * w[3].tanh();
    let l02 = 0.8 * w[4].tanh();
    let l12 = 0.8 * w[5].tanh();
    let v = [
        l00 * d[0] + l01 * d[1] + l02 * d[2],
        l11 * d[1] + l12 * d[2],
        l22 * d[2],
    ];
    let n = (dot3(v, v) + EPS * dot3(d, d)).sqrt().max(1e-12);
    // L^T v + eps d
    let ltv = [
        l00 * v[0] + EPS * d[0],
        l01 * v[0] + l11 * v[1] + EPS * d[1],
        l02 * v[0] + l12 * v[1] + l22 * v[2] + EPS * d[2],
    ];
    (n, [ltv[0] / n, ltv[1] / n, ltv[2] / n])
}

fn area(p: [V; 3], w: &[f64]) -> f64 {
    let u = sub(p[1], p[0]);
    let v = sub(p[2], p[0]);
    0.5 * (inner(u, u, w) * inner(v, v, w) - inner(u, v, w).powi(2))
        .max(0.0)
        .sqrt()
}
fn volume(p: [V; 4], w: &[f64]) -> f64 {
    let d = [sub(p[1], p[0]), sub(p[2], p[0]), sub(p[3], p[0])];
    let gram: [V; 3] = std::array::from_fn(|i| std::array::from_fn(|j| inner(d[i], d[j], w)));
    (det3(gram[0], gram[1], gram[2]).max(0.0)).sqrt() / 6.0
}
fn determinant(mut a: Vec<Vec<f64>>) -> f64 {
    let mut d = 1.0;
    for j in 0..a.len() {
        let k = (j..a.len())
            .max_by(|&i, &k| a[i][j].abs().total_cmp(&a[k][j].abs()))
            .unwrap();
        if a[k][j].abs() < 1e-15 {
            return 0.0;
        }
        if k != j {
            a.swap(j, k);
            d = -d;
        }
        let z = a[j][j];
        d *= z;
        for i in j + 1..a.len() {
            let r = a[i][j] / z;
            for k in j + 1..a.len() {
                a[i][k] -= r * a[j][k];
            }
        }
    }
    d
}
fn cm_volume2(p: [V; 4], w: &[f64]) -> f64 {
    let mut a = vec![vec![0.0; 5]; 5];
    for i in 0..4 {
        a[0][i + 1] = 1.0;
        a[i + 1][0] = 1.0;
        for j in 0..4 {
            let d = sub(p[i], p[j]);
            a[i + 1][j + 1] = inner(d, d, w);
        }
    }
    determinant(a) / 288.0
}

/// Base no lineal de 9 características de Fourier-Polinomios para la cara f = [1, a, b]
pub fn face_nonlinear_basis(f: [f64; 3]) -> [f64; 9] {
    let a = f[1];
    let b = f[2];
    let da = a - 0.5;
    let db = b - 0.5;
    [
        1.0,
        a,
        b,
        da * da,
        db * db,
        da * db,
        (7.0 * a).sin(),
        (7.0 * a).cos(),
        (7.0 * b).sin(),
    ]
}

#[derive(Clone)]
pub struct Mesh {
    pub p: Vec<V>,
    pub terrain: Vec<[f64; 2]>,
    pub tets: Vec<[usize; 4]>,
    pub faces: Vec<[usize; 3]>,
    pub edges: Vec<[usize; 2]>,
    pub b2: Vec<Vec<(usize, f64)>>,
    pub b3: Vec<Vec<(usize, f64)>>,
    pub face_tets: Vec<Vec<usize>>,
    pub face_features: Vec<[f64; 3]>,
    pub face_basis9: Vec<[f64; 9]>,
    pub edge_hodge_down: Vec<f64>,
    pub edge_hodge_up: Vec<f64>,
    pub face_holonomy_b: Vec<f64>,
}

impl Mesh {
    pub fn new(p: Vec<V>, terrain: Vec<[f64; 2]>, mut tets: Vec<[usize; 4]>) -> Self {
        for t in &mut tets {
            t.sort_unstable();
        }
        tets.sort_unstable();
        tets.dedup();
        let mut fs = BTreeSet::new();
        let mut es = BTreeSet::new();
        for t in &tets {
            for i in 0..4 {
                let f: Vec<usize> = (0..4).filter(|&j| j != i).map(|j| t[j]).collect();
                fs.insert([f[0], f[1], f[2]]);
            }
            for i in 0..4 {
                for j in i + 1..4 {
                    es.insert([t[i], t[j]]);
                }
            }
        }
        let faces: Vec<_> = fs.into_iter().collect();
        let edges: Vec<_> = es.into_iter().collect();
        let fi: BTreeMap<_, _> = faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
        let ei: BTreeMap<_, _> = edges.iter().enumerate().map(|(i, &e)| (e, i)).collect();
        let b2: Vec<_> = faces
            .iter()
            .map(|f| {
                vec![
                    (ei[&[f[1], f[2]]], 1.0),
                    (ei[&[f[0], f[2]]], -1.0),
                    (ei[&[f[0], f[1]]], 1.0),
                ]
            })
            .collect();
        let b3: Vec<Vec<(usize, f64)>> = tets
            .iter()
            .map(|t| {
                (0..4)
                    .map(|j| {
                        let f: Vec<usize> = (0..4).filter(|&k| k != j).map(|k| t[k]).collect();
                        (fi[&[f[0], f[1], f[2]]], if j % 2 == 0 { 1.0 } else { -1.0 })
                    })
                    .collect()
            })
            .collect();
        let mut face_tets = vec![vec![]; faces.len()];
        for (t, fs) in b3.iter().enumerate() {
            for &(f, _) in fs {
                face_tets[f].push(t);
            }
        }
        let face_features: Vec<[f64; 3]> = faces
            .iter()
            .map(|f| {
                [
                    1.0,
                    f.iter().map(|&x| terrain[x][0]).sum::<f64>() / 3.0,
                    f.iter().map(|&x| terrain[x][1]).sum::<f64>() / 3.0,
                ]
            })
            .collect();
        let face_basis9: Vec<[f64; 9]> = face_features
            .iter()
            .map(|&f| face_nonlinear_basis(f))
            .collect();

        // 1-cocadena antisimétrica de Poincaré A_0([v_i, v_j]) = 0.5 * (a_i b_j - a_j b_i)
        // Su coborde B = d_1 A_0 satisface identidad de Bianchi d_2 B = d_2 d_1 A_0 = 0 en cada tetraedro
        let a_cochain: Vec<f64> = edges
            .iter()
            .map(|e| {
                let (i, j) = (e[0], e[1]);
                0.5 * (terrain[i][0] * terrain[j][1] - terrain[j][0] * terrain[i][1])
            })
            .collect();
        let face_holonomy_b: Vec<f64> = b2
            .iter()
            .map(|b| b.iter().map(|&(eid, s)| s * a_cochain[eid]).sum())
            .collect();

        let mut m = Self {
            p,
            terrain,
            tets,
            faces,
            edges,
            b2,
            b3,
            face_tets,
            face_features,
            face_basis9,
            edge_hodge_down: Vec::new(),
            edge_hodge_up: Vec::new(),
            face_holonomy_b,
        };

        // Precomputar descomposición espectral Hodge-1 (\Delta_1^{down}, \Delta_1^{up}) sobre el gradiente de terreno
        let w_ref = [0.0; P];
        let (m0, m1, m2) = m.hodge_weights(&w_ref);
        let h0: Vec<f64> = m
            .edges
            .iter()
            .map(|e| {
                (m.terrain[e[1]][0] - m.terrain[e[0]][0])
                    + 0.7 * (m.terrain[e[1]][1] - m.terrain[e[0]][1])
            })
            .collect();
        let (h_down, h_up) = m.hodge1_split(&h0, &m0, &m1, &m2);
        let norm_down = h_down
            .iter()
            .map(|x| x.abs())
            .fold(1e-6_f64, f64::max);
        let norm_up = h_up.iter().map(|x| x.abs()).fold(1e-6_f64, f64::max);
        m.edge_hodge_down = h_down.into_iter().map(|x| (x / norm_down).clamp(-1.0, 1.0)).collect();
        m.edge_hodge_up = h_up.into_iter().map(|x| (x / norm_up).clamp(-1.0, 1.0)).collect();
        m
    }

    pub fn from_external(ext: &ExternalSimplicialFile) -> Self {
        Self::new(
            ext.vertices.clone(),
            ext.terrain.clone(),
            ext.tetrahedra.clone(),
        )
    }

    fn generate(r: &mut Rng, dims: [usize; 3], jitter: f64) -> Self {
        loop {
            let [nx, ny, nz] = dims;
            let id = |x: usize, y: usize, z: usize| x + (nx + 1) * (y + (ny + 1) * z);
            let mut base = vec![];
            for z in 0..=nz {
                for y in 0..=ny {
                    for x in 0..=nx {
                        base.push([x as f64, y as f64, z as f64]);
                    }
                }
            }
            let p: Vec<V> = base
                .iter()
                .map(|b| std::array::from_fn(|i| b[i] + jitter * (2.0 * r.uniform() - 1.0)))
                .collect();
            let terrain = (0..p.len()).map(|_| [r.uniform(), r.uniform()]).collect();
            let perms = [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ];
            let mut tets = vec![];
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        for perm in perms {
                            let mut c = [x, y, z];
                            let mut t = [id(x, y, z), 0, 0, 0];
                            for k in 0..3 {
                                c[perm[k]] += 1;
                                t[k + 1] = id(c[0], c[1], c[2]);
                            }
                            if r.uniform() >= 0.18 {
                                tets.push(t);
                            }
                        }
                    }
                }
            }
            if tets.is_empty() {
                continue;
            }
            if tets.iter().any(|t| {
                let a = sub(p[t[1]], p[t[0]]);
                let b = sub(p[t[2]], p[t[0]]);
                let c = sub(p[t[3]], p[t[0]]);
                let aa = sub(base[t[1]], base[t[0]]);
                let bb = sub(base[t[2]], base[t[0]]);
                let cc = sub(base[t[3]], base[t[0]]);
                det3(a, b, c) * det3(aa, bb, cc) <= 0.05
            }) {
                continue;
            }
            if (0..p.len()).any(|i| !tets.iter().any(|t| t.contains(&i))) {
                continue;
            }
            let m = Self::new(p, terrain, tets);
            let g = Graph::new(&m, false, false);
            if g.connected() {
                return m;
            }
        }
    }

    pub fn boundary_checks(&self) -> bool {
        for f in &self.b2 {
            let mut z = vec![0.0; self.p.len()];
            for &(e, s) in f {
                z[self.edges[e][0]] -= s;
                z[self.edges[e][1]] += s;
            }
            if z.iter().any(|x| *x != 0.0) {
                return false;
            }
        }
        for t in &self.b3 {
            let mut z = vec![0.0; self.edges.len()];
            for &(f, s) in t {
                for &(e, q) in &self.b2[f] {
                    z[e] += s * q;
                }
            }
            if z.iter().any(|x| *x != 0.0) {
                return false;
            }
        }
        true
    }

    pub fn hodge_weights(&self, w: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut a = vec![0.0; self.p.len()];
        let mut b = vec![0.0; self.edges.len()];
        let mut c = vec![0.0; self.faces.len()];
        for (tid, t) in self.tets.iter().enumerate() {
            let vol = volume(t.map(|v| self.p[v]), w);
            for &v in t {
                a[v] += vol / 4.0;
            }
            for (eid, e) in self.edges.iter().enumerate() {
                if e.iter().all(|v| t.contains(v)) {
                    let d = sub(self.p[e[1]], self.p[e[0]]);
                    b[eid] += vol / (6.0 * inner(d, d, w));
                }
            }
            for &(f, _) in &self.b3[tid] {
                let ar = area(self.faces[f].map(|v| self.p[v]), w);
                c[f] += vol / (4.0 * ar * ar);
            }
        }
        (a, b, c)
    }

    pub fn hodge1_split(
        &self,
        h: &[f64],
        m0: &[f64],
        m1: &[f64],
        m2: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let mut div = vec![0.0; self.p.len()];
        for (i, e) in self.edges.iter().enumerate() {
            div[e[0]] -= m1[i] * h[i];
            div[e[1]] += m1[i] * h[i];
        }
        for i in 0..div.len() {
            div[i] /= m0[i].max(1e-15);
        }
        let down: Vec<f64> = self.edges.iter().map(|e| div[e[1]] - div[e[0]]).collect();
        let curl: Vec<f64> = self
            .b2
            .iter()
            .enumerate()
            .map(|(f, b)| m2[f] * b.iter().map(|&(e, s)| s * h[e]).sum::<f64>())
            .collect();
        let mut up = vec![0.0; self.edges.len()];
        for (f, b) in self.b2.iter().enumerate() {
            for &(e, s) in b {
                up[e] += s * curl[f] / m1[e].max(1e-15);
            }
        }
        (down, up)
    }

    pub fn hodge1(&self, h: &[f64], m0: &[f64], m1: &[f64], m2: &[f64]) -> Vec<f64> {
        let (down, up) = self.hodge1_split(h, m0, m1, m2);
        down.into_iter().zip(up).map(|(a, b)| a + b).collect()
    }

    pub fn regge(&self, w: &[f64]) -> (usize, f64, f64) {
        let mut count = 0;
        let mut energy = 0.0;
        let mut max: f64 = 0.0;
        for e in &self.edges {
            if self
                .faces
                .iter()
                .enumerate()
                .any(|(i, f)| self.face_tets[i].len() == 1 && e.iter().all(|v| f.contains(v)))
            {
                continue;
            }
            let mut angles = 0.0;
            for t in &self.tets {
                if !e.iter().all(|v| t.contains(v)) {
                    continue;
                }
                let other: Vec<_> = t.iter().filter(|v| !e.contains(v)).copied().collect();
                let v = sub(self.p[e[1]], self.p[e[0]]);
                let u = sub(self.p[other[0]], self.p[e[0]]);
                let z = sub(self.p[other[1]], self.p[e[0]]);
                let pu = sub(u, scale(v, inner(u, v, w) / inner(v, v, w)));
                let pz = sub(z, scale(v, inner(z, v, w) / inner(v, v, w)));
                let cos = inner(pu, pz, w) / (inner(pu, pu, w) * inner(pz, pz, w)).sqrt();
                angles += cos.clamp(-1.0, 1.0).acos();
            }
            let deficit = std::f64::consts::TAU - angles;
            count += 1;
            max = max.max(deficit.abs());
            energy += length(sub(self.p[e[1]], self.p[e[0]]), w).0 * deficit * deficit;
        }
        (count, energy, max)
    }

    /// Calcula las longitudes de arista intrínsecas deformadas por Hodge:
    /// \ell_e(\theta) = \ell_e^{SPD}(\theta) * \exp(0.15 \tanh(w_{15}) h_{down}(e) + 0.15 \tanh(w_{16}) h_{up}(e))
    pub fn intrinsic_edge_lengths_v3(&self, w18: &[f64]) -> Vec<f64> {
        let w_down = if w18.len() > 15 { 0.15 * w18[15].tanh() } else { 0.0 };
        let w_up = if w18.len() > 16 { 0.15 * w18[16].tanh() } else { 0.0 };
        self.edges
            .iter()
            .enumerate()
            .map(|(eid, e)| {
                let d = sub(self.p[e[1]], self.p[e[0]]);
                let l_spd = inner(d, d, w18).sqrt();
                let stretch = (w_down * self.edge_hodge_down[eid] + w_up * self.edge_hodge_up[eid]).exp();
                (l_spd * stretch).max(1e-8)
            })
            .collect()
    }

    /// Calcula la acción de curvatura intrínseca de Regge S_Regge(\ell_\theta), la barrera de Cayley-Menger
    /// B_CM(\ell_\theta) y la energía armónica/Dirichlet de Hodge \mathcal{E}_{Hodge}(\theta) para entrenamiento v3.
    pub fn intrinsic_regge_cm_hodge_v3(&self, w18: &[f64]) -> (f64, f64, f64, f64, f64) {
        let edge_lengths = self.intrinsic_edge_lengths_v3(w18);
        let edge_map: BTreeMap<[usize; 2], usize> = self
            .edges
            .iter()
            .enumerate()
            .map(|(i, &e)| (e, i))
            .collect();
        let get_l2 = |a: usize, b: usize| -> f64 {
            if a == b {
                return 0.0;
            }
            let key = if a < b { [a, b] } else { [b, a] };
            let l = edge_lengths[edge_map[&key]];
            l * l
        };

        // 1. Curvatura intrínseca de Regge sobre bisagras interiores
        let mut s_regge = 0.0;
        let mut max_deficit = 0.0f64;
        for (eid, e) in self.edges.iter().enumerate() {
            if self
                .faces
                .iter()
                .enumerate()
                .any(|(i, f)| self.face_tets[i].len() == 1 && e.iter().all(|v| f.contains(v)))
            {
                continue;
            }
            let mut angles = 0.0;
            for t in &self.tets {
                if !e.iter().all(|v| t.contains(v)) {
                    continue;
                }
                let other: Vec<usize> = t.iter().filter(|v| !e.contains(v)).copied().collect();
                let (v0, v1, v2, v3) = (e[0], e[1], other[0], other[1]);
                let l01_2 = get_l2(v0, v1);
                let l02_2 = get_l2(v0, v2);
                let l03_2 = get_l2(v0, v3);
                let l12_2 = get_l2(v1, v2);
                let l13_2 = get_l2(v1, v3);
                let l23_2 = get_l2(v2, v3);

                let g11 = l01_2.max(1e-12);
                let g22 = l02_2;
                let g33 = l03_2;
                let g12 = 0.5 * (l01_2 + l02_2 - l12_2);
                let g13 = 0.5 * (l01_2 + l03_2 - l13_2);
                let g23 = 0.5 * (l02_2 + l03_2 - l23_2);

                let p22 = (g22 - (g12 * g12) / g11).max(1e-12);
                let p33 = (g33 - (g13 * g13) / g11).max(1e-12);
                let p23 = g23 - (g12 * g13) / g11;
                let cos_theta = (p23 / (p22 * p33).sqrt()).clamp(-1.0 + 1e-12, 1.0 - 1e-12);
                angles += cos_theta.acos();
            }
            let deficit = std::f64::consts::TAU - angles;
            max_deficit = max_deficit.max(deficit.abs());
            s_regge += edge_lengths[eid] * deficit * deficit;
        }

        // 2. Determinante de Cayley-Menger intrínseco y barrera logarítmica B_CM
        let mut b_cm = 0.0;
        let mut min_cm_vol2 = f64::INFINITY;
        for t in &self.tets {
            let mut cm = vec![vec![0.0; 5]; 5];
            for i in 0..4 {
                cm[0][i + 1] = 1.0;
                cm[i + 1][0] = 1.0;
                for j in 0..4 {
                    cm[i + 1][j + 1] = get_l2(t[i], t[j]);
                }
            }
            let vol2 = (determinant(cm) / 288.0).max(1e-8);
            min_cm_vol2 = min_cm_vol2.min(vol2);
            b_cm += -vol2.ln() / (self.tets.len() as f64);
        }

        // 3. Energía de Dirichlet-Hodge \langle h_\theta, \Delta_1(\theta) h_\theta \rangle_{M_1}
        let (m0, m1, m2) = self.hodge_weights(w18);
        let c_down = if w18.len() > 15 { w18[15].tanh() } else { 0.0 };
        let c_up = if w18.len() > 16 { w18[16].tanh() } else { 0.0 };
        let h_theta: Vec<f64> = (0..self.edges.len())
            .map(|i| c_down * self.edge_hodge_down[i] + c_up * self.edge_hodge_up[i])
            .collect();
        let lh = self.hodge1(&h_theta, &m0, &m1, &m2);
        let e_hodge: f64 = (0..self.edges.len())
            .map(|i| h_theta[i] * m1[i] * lh[i])
            .sum::<f64>()
            / (self.edges.len() as f64);

        (s_regge, b_cm, e_hodge, max_deficit, min_cm_vol2)
    }
}

#[derive(Clone)]
pub struct Point {
    pub p: V,
    pub support: Vec<usize>,
}
#[derive(Clone)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub d: V,
    pub faces: Vec<usize>,
    pub mesh_edges: Vec<usize>,
    pub carrier_dim: usize,
    pub orientation_sign: f64,
}
#[derive(Clone)]
pub struct Graph {
    pub points: Vec<Point>,
    pub edges: Vec<Edge>,
    pub out: Vec<Vec<usize>>,
    pub n_original: usize,
}

impl Graph {
    pub fn new(m: &Mesh, edge_only: bool, refine: bool) -> Self {
        Self::new_with_level(m, edge_only, if refine { 1 } else { 0 })
    }

    pub fn new_with_level(m: &Mesh, edge_only: bool, refine_level: usize) -> Self {
        let mut points: Vec<_> = m
            .p
            .iter()
            .enumerate()
            .map(|(i, &p)| Point {
                p,
                support: vec![i],
            })
            .collect();
        let nv = points.len();
        if !edge_only {
            for f in &m.faces {
                points.push(Point {
                    p: scale(f.iter().fold([0.0; 3], |a, &v| add(a, m.p[v])), 1.0 / 3.0),
                    support: f.to_vec(),
                });
            }
            for t in &m.tets {
                points.push(Point {
                    p: scale(t.iter().fold([0.0; 3], |a, &v| add(a, m.p[v])), 0.25),
                    support: t.to_vec(),
                });
            }
            if refine_level >= 1 {
                for e in &m.edges {
                    points.push(Point {
                        p: scale(add(m.p[e[0]], m.p[e[1]]), 0.5),
                        support: e.to_vec(),
                    });
                }
                for f in &m.faces {
                    for k in 0..3 {
                        points.push(Point {
                            p: std::array::from_fn(|d| {
                                f.iter()
                                    .enumerate()
                                    .map(|(j, &v)| {
                                        m.p[v][d] * if j == k { 2.0 / 3.0 } else { 1.0 / 6.0 }
                                    })
                                    .sum()
                            }),
                            support: f.to_vec(),
                        });
                    }
                }
            }
            if refine_level >= 2 {
                // Nivel 2 ultra-fino para cuantificar convergencia continua:
                // cuartos de arista (0.25, 0.75) y puntos baricéntricos secundarios de cara (0.15, 0.425, 0.425)
                for e in &m.edges {
                    for &alpha in &[0.25, 0.75] {
                        points.push(Point {
                            p: add(scale(m.p[e[0]], alpha), scale(m.p[e[1]], 1.0 - alpha)),
                            support: e.to_vec(),
                        });
                    }
                }
                for f in &m.faces {
                    for k in 0..3 {
                        points.push(Point {
                            p: std::array::from_fn(|d| {
                                f.iter()
                                    .enumerate()
                                    .map(|(j, &v)| {
                                        m.p[v][d] * if j == k { 0.14 } else { 0.43 }
                                    })
                                    .sum()
                            }),
                            support: f.to_vec(),
                        });
                    }
                }
            }
        }
        let mut pairs = BTreeSet::new();
        for t in &m.tets {
            let ids: Vec<_> = points
                .iter()
                .enumerate()
                .filter(|(_, p)| p.support.iter().all(|v| t.contains(v)))
                .map(|(i, _)| i)
                .collect();
            for (j, &a) in ids.iter().enumerate() {
                for &b in &ids[j + 1..] {
                    pairs.insert((a, b));
                }
            }
        }
        let mut edges = vec![];
        let mut out = vec![vec![]; points.len()];
        for (a, b) in pairs {
            let mut support = points[a].support.clone();
            support.extend(&points[b].support);
            support.sort_unstable();
            support.dedup();
            let carrier_dim = support.len() - 1;
            let faces: Vec<usize> = m
                .faces
                .iter()
                .enumerate()
                .filter(|(_, f)| {
                    if support.len() == 2 {
                        support.iter().all(|v| f.contains(v))
                    } else if support.len() == 3 {
                        f.as_slice() == support.as_slice()
                    } else {
                        f.iter().all(|v| support.contains(v))
                    }
                })
                .map(|(i, _)| i)
                .collect();
            assert!(!faces.is_empty());
            let mesh_edges: Vec<usize> = m
                .edges
                .iter()
                .enumerate()
                .filter(|(_, me)| me.iter().all(|v| support.contains(v)))
                .map(|(i, _)| i)
                .collect();
            for (src, dst, sign) in [(a, b, 1.0), (b, a, -1.0)] {
                out[src].push(edges.len());
                edges.push(Edge {
                    src,
                    dst,
                    d: sub(points[dst].p, points[src].p),
                    faces: faces.clone(),
                    mesh_edges: mesh_edges.clone(),
                    carrier_dim,
                    orientation_sign: sign,
                });
            }
        }
        Self {
            points,
            edges,
            out,
            n_original: nv,
        }
    }

    pub fn connected(&self) -> bool {
        if self.points.is_empty() {
            return false;
        }
        let mut vis = vec![false; self.points.len()];
        let mut q = vec![0];
        vis[0] = true;
        while let Some(x) = q.pop() {
            for &id in &self.out[x] {
                let y = self.edges[id].dst;
                if !vis[y] {
                    vis[y] = true;
                    q.push(y);
                }
            }
        }
        vis.iter().all(|x| *x)
    }

    pub fn distances(&self, c: &[f64], goal: usize) -> Vec<f64> {
        let n = self.points.len();
        let mut d = vec![f64::INFINITY; n];
        d[goal] = 0.0;
        let mut used = vec![false; n];
        for _ in 0..n {
            let x = (0..n)
                .filter(|&x| !used[x])
                .min_by(|&a, &b| d[a].total_cmp(&d[b]))
                .unwrap();
            if !d[x].is_finite() {
                break;
            }
            used[x] = true;
            for &id in &self.out[x] {
                let e = &self.edges[id];
                let reverse = self.out[e.dst]
                    .iter()
                    .find(|&&j| self.edges[j].dst == x)
                    .unwrap();
                d[e.dst] = d[e.dst].min(d[x] + c[*reverse]);
            }
        }
        d
    }

    pub fn targets(&self, c: &[f64], d: &[f64], goal: usize) -> Vec<usize> {
        (0..self.points.len())
            .map(|x| {
                if x == goal || !d[x].is_finite() {
                    usize::MAX
                } else {
                    *self.out[x]
                        .iter()
                        .min_by(|&&a, &&b| {
                            (c[a] + d[self.edges[a].dst]).total_cmp(&(c[b] + d[self.edges[b].dst]))
                        })
                        .unwrap()
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Full,
    Diagonal,
    EdgeDiagonal,
    NoPhi,
    Untrained,
    Euclidean,
    Oracle,
    V3FullGpu,
    V3ContinuousGpu,
    V3ZeroShotGpu,
    V3AdaptedGpu,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Diagonal => "diagonal",
            Self::EdgeDiagonal => "edge_diagonal",
            Self::NoPhi => "no_phi",
            Self::Untrained => "untrained",
            Self::Euclidean => "euclidean",
            Self::Oracle => "oracle",
            Self::V3FullGpu => "v3_full_gpu",
            Self::V3ContinuousGpu => "v3_continuous_gpu",
            Self::V3ZeroShotGpu => "v3_zeroshot_gpu",
            Self::V3AdaptedGpu => "v3_adapted_gpu",
        }
    }
    pub fn trainable(self, j: usize) -> bool {
        match self {
            Self::Diagonal | Self::EdgeDiagonal => j < 3 || (j >= 6 && j < 9),
            Self::NoPhi => j < 6,
            Self::Full => j < 9,
            Self::V3FullGpu | Self::V3ContinuousGpu | Self::V3AdaptedGpu => j < P_V3,
            _ => false,
        }
    }
}

const TRAIN_KINDS: [Kind; 4] = [Kind::Full, Kind::Diagonal, Kind::EdgeDiagonal, Kind::NoPhi];

pub fn costs(m: &Mesh, g: &Graph, w: &[f64], kind: Kind, changed: bool) -> (Vec<f64>, Vec<[f64; P]>) {
    let mut ww = vec![0.0; P];
    for j in 0..P.min(w.len()) {
        ww[j] = w[j];
    }
    if kind == Kind::Diagonal || kind == Kind::EdgeDiagonal {
        ww[3..6].fill(0.0);
    }
    let phi: Vec<(f64, [f64; 3])> = m
        .face_features
        .iter()
        .map(|f| {
            let a = dot(&ww[6..9], f);
            let factor = if changed {
                1.0 + 0.45 * (7.0 * f[1]).sin() + 0.8 * (f[2] - 0.5).powi(2)
            } else {
                1.0
            };
            (
                factor * (0.3 + sp(a)),
                std::array::from_fn(|j| sig(a) * f[j] * factor),
            )
        })
        .collect();
    let mut c = vec![];
    let mut dc = vec![];
    for e in &g.edges {
        if kind == Kind::Euclidean {
            c.push(dot3(e.d, e.d).sqrt());
            dc.push([0.0; P]);
            continue;
        }
        let (n, mut grad) = length(e.d, &ww);
        let imp = if kind == Kind::NoPhi {
            1.0
        } else {
            e.faces.iter().map(|&f| phi[f].0).sum::<f64>() / e.faces.len() as f64
        };
        for j in 0..6 {
            grad[j] *= imp;
        }
        if kind != Kind::NoPhi {
            for j in 0..3 {
                grad[6 + j] =
                    n * e.faces.iter().map(|&f| phi[f].1[j]).sum::<f64>() / e.faces.len() as f64;
            }
        }
        for j in 0..P {
            if !kind.trainable(j) && kind != Kind::Untrained && kind != Kind::Oracle {
                grad[j] = 0.0;
            }
        }
        c.push(n * imp);
        dc.push(grad);
    }
    assert!(c.iter().all(|v| v.is_finite() && *v > 0.0));
    (c, dc)
}

/// Evaluación de costos y gradientes analíticos para el modelo ampliado v3 de 18 parámetros:
/// - w[0..6]: Métrica SPD 3x3 global L L^T + \eps I
/// - w[6..15]: Cabeza no lineal de 9 funciones base Fourier-Polinomiales \Phi_\theta
/// - w[15..17]: Modulación intrínseca de longitud acoplada al Laplaciano de Hodge \Delta_1 (down + up)
/// - w[17]: Flujo magnético / holonomía de coborde 2-cocadena B = d_1 A
pub fn costs_v3(
    m: &Mesh,
    g: &Graph,
    w18: &[f64],
    epistemic_gate: bool,
) -> (Vec<f64>, Vec<[f64; P_V3]>) {
    let mut ww = [0.0; P_V3];
    for (j, &val) in w18.iter().take(P_V3).enumerate() {
        ww[j] = val;
    }

    // Si se activa el gate epistémico (transferencia zero-shot ante cambio de ley del terreno),
    // neutraliza el cabezal superficial descalibrado y enruta mediante el tensor métrico SPD aprendido G_\theta.
    let gate_scale = if epistemic_gate { 0.0 } else { 1.0 };

    let phi_faces: Vec<(f64, [f64; 9])> = m
        .face_basis9
        .iter()
        .map(|basis| {
            let mut z = ww[6] * basis[0];
            for k in 1..9 {
                z += gate_scale * ww[6 + k] * basis[k];
            }
            let val = 0.3 + sp(z);
            let s = sig(z) * gate_scale;
            let mut dphi = [0.0; 9];
            dphi[0] = sig(z) * basis[0];
            for k in 1..9 {
                dphi[k] = s * basis[k];
            }
            (val, dphi)
        })
        .collect();

    let th15 = ww[15].tanh();
    let th16 = ww[16].tanh();
    let th17 = ww[17].tanh();
    let sech2_15 = 1.0 - th15 * th15;
    let sech2_16 = 1.0 - th16 * th16;
    let sech2_17 = 1.0 - th17 * th17;

    let mut c = Vec::with_capacity(g.edges.len());
    let mut dc = Vec::with_capacity(g.edges.len());

    for e in &g.edges {
        let (n_spd, grad_spd) = length(e.d, &ww[0..9]);

        let (h_down, h_up) = if e.mesh_edges.is_empty() {
            (0.0, 0.0)
        } else {
            let inv = 1.0 / (e.mesh_edges.len() as f64);
            let hd = e.mesh_edges.iter().map(|&me| m.edge_hodge_down[me]).sum::<f64>() * inv;
            let hu = e.mesh_edges.iter().map(|&me| m.edge_hodge_up[me]).sum::<f64>() * inv;
            (hd, hu)
        };

        let hodge_stretch = (0.15 * th15 * h_down + 0.15 * th16 * h_up).exp();
        let inv_f = 1.0 / (e.faces.len() as f64);
        let imp = e.faces.iter().map(|&f| phi_faces[f].0).sum::<f64>() * inv_f;

        // Holonomía simétrica/direccional de 2-cocadena B = d_1 A
        let b_hol_mean = e.faces.iter().map(|&f| m.face_holonomy_b[f]).sum::<f64>() * inv_f;
        let b_term = e.orientation_sign * b_hol_mean;
        let hol_factor = (0.10 * th17 * b_term).exp();

        let edge_cost = n_spd * hodge_stretch * imp * hol_factor;
        let mut grad = [0.0; P_V3];

        // Gradientes respecto a w[0..6] (SPD)
        for j in 0..6 {
            grad[j] = grad_spd[j] * hodge_stretch * imp * hol_factor;
        }
        // Gradientes respecto a w[6..15] (Fourier-Polinomial Phi)
        for k in 0..9 {
            let d_imp = e.faces.iter().map(|&f| phi_faces[f].1[k]).sum::<f64>() * inv_f;
            grad[6 + k] = n_spd * hodge_stretch * d_imp * hol_factor;
        }
        // Gradientes respecto a w[15..17] (Hodge down & up)
        grad[15] = edge_cost * 0.15 * sech2_15 * h_down;
        grad[16] = edge_cost * 0.15 * sech2_16 * h_up;
        // Gradiente respecto a w[17] (Holonomía B = d_1 A)
        grad[17] = edge_cost * 0.10 * sech2_17 * b_term;

        c.push(edge_cost);
        dc.push(grad);
    }

    (c, dc)
}

pub const TEACHER: [f64; P] = [0.15, -0.15, 0.05, 1.2, -0.95, 0.8, -0.7, 2.0, 1.2];

#[derive(Clone)]
pub struct Case {
    pub m: Mesh,
    pub g: Graph,
    pub eg: Graph,
    pub rg: Graph,
    pub goal: usize,
    pub truth: Vec<f64>,
    pub etruth: Vec<f64>,
    pub rtruth: Vec<f64>,
    pub dist: Vec<f64>,
    pub edist: Vec<f64>,
    pub rdist: Vec<f64>,
    pub target: Vec<usize>,
    pub etarget: Vec<usize>,
    pub rtarget: Vec<usize>,
    pub changed: bool,
}

impl Case {
    pub fn new(m: Mesh, goal: usize, changed: bool) -> Self {
        let g = Graph::new(&m, false, false);
        let eg = Graph::new(&m, true, false);
        let rg = Graph::new(&m, false, true);
        let truth = costs(&m, &g, &TEACHER, Kind::Full, changed).0;
        let etruth = costs(&m, &eg, &TEACHER, Kind::Full, changed).0;
        let rtruth = costs(&m, &rg, &TEACHER, Kind::Full, changed).0;
        let dist = g.distances(&truth, goal);
        let edist = eg.distances(&etruth, goal);
        let rdist = rg.distances(&rtruth, goal);
        let target = g.targets(&truth, &dist, goal);
        let etarget = eg.targets(&etruth, &edist, goal);
        let rtarget = rg.targets(&rtruth, &rdist, goal);
        Self {
            m,
            g,
            eg,
            rg,
            goal,
            truth,
            etruth,
            rtruth,
            dist,
            edist,
            rdist,
            target,
            etarget,
            rtarget,
            changed,
        }
    }
    pub fn graph(&self, k: Kind) -> &Graph {
        match k {
            Kind::EdgeDiagonal => &self.eg,
            Kind::V3ContinuousGpu => &self.rg,
            _ => &self.g,
        }
    }
}

pub struct Tape {
    pub v: Vec<f64>,
    pub p: Vec<f64>,
    pub n: usize,
    pub h: usize,
}

pub fn forward(g: &Graph, c: &[f64], goal: usize) -> Tape {
    let n = g.points.len();
    let h = n;
    let ne = g.edges.len();
    let mut v = vec![f64::INFINITY; (h + 1) * n];
    let mut p = vec![0.0; (h + 1) * ne];
    v[goal] = 0.0;
    for t in 1..=h {
        v[t * n + goal] = 0.0;
        for x in 0..n {
            if x == goal {
                continue;
            }
            let q: Vec<_> = g.out[x]
                .iter()
                .map(|&id| c[id] + v[(t - 1) * n + g.edges[id].dst])
                .collect();
            let (val, pr) = softmin(&q);
            v[t * n + x] = val;
            for (&id, prob) in g.out[x].iter().zip(pr) {
                p[t * ne + id] = prob;
            }
        }
    }
    Tape { v, p, n, h }
}

pub fn loss(case: &Case, w: &[f64], kind: Kind, gradient: bool) -> (f64, Vec<f64>) {
    let g = case.graph(kind);
    let target = if kind == Kind::EdgeDiagonal {
        &case.etarget
    } else {
        &case.target
    };
    let (c, dc) = costs(&case.m, g, w, kind, false);
    let t = forward(g, &c, case.goal);
    let ne = g.edges.len();
    let count = (t.n - 1) as f64;
    let mut l = 0.0;
    let mut gc = vec![0.0; ne];
    let mut av = vec![0.0; t.v.len()];
    for x in 0..t.n {
        if x == case.goal {
            continue;
        }
        let tar = target[x];
        let y = g.edges[tar].dst;
        l += (c[tar] + t.v[(t.h - 1) * t.n + y] - t.v[t.h * t.n + x]) / TAU / count;
        if gradient {
            for &id in &g.out[x] {
                let q = (f64::from(id == tar) - t.p[t.h * ne + id]) / TAU / count;
                gc[id] += q;
                av[(t.h - 1) * t.n + g.edges[id].dst] += q;
            }
        }
    }
    if gradient {
        for layer in (1..t.h).rev() {
            for x in 0..t.n {
                if x == case.goal {
                    continue;
                }
                let adj = av[layer * t.n + x];
                if adj == 0.0 {
                    continue;
                }
                for &id in &g.out[x] {
                    let d = adj * t.p[layer * ne + id];
                    gc[id] += d;
                    av[(layer - 1) * t.n + g.edges[id].dst] += d;
                }
            }
        }
    }
    let mut grad = vec![0.0; P];
    if gradient {
        for id in 0..ne {
            for j in 0..P {
                grad[j] += gc[id] * dc[id][j];
            }
        }
    }
    assert!(l.is_finite());
    (l, grad)
}

/// Funcional de Energía Libre completo de v3 con regularizadores geométricos entrenados:
/// L(w) = L_{route}(V_\tau^*(w)) + \alpha S_{Regge}(\ell_w) + \beta B_{CM}(\ell_w) + \gamma E_{Hodge}(w) + \lambda_{fourier} \|w_{9..15}\|^2
pub fn loss_v3(
    case: &Case,
    w18: &[f64],
    alpha_regge: f64,
    beta_cm: f64,
    gamma_hodge: f64,
    fourier_l2: f64,
    gradient: bool,
) -> (f64, Vec<f64>) {
    let g = &case.g;
    let target = &case.target;
    let (c, dc) = costs_v3(&case.m, g, w18, false);
    let t = forward(g, &c, case.goal);
    let ne = g.edges.len();
    let count = (t.n - 1) as f64;
    let mut l_route = 0.0;
    let mut gc = vec![0.0; ne];
    let mut av = vec![0.0; t.v.len()];

    for x in 0..t.n {
        if x == case.goal {
            continue;
        }
        let tar = target[x];
        let y = g.edges[tar].dst;
        l_route += (c[tar] + t.v[(t.h - 1) * t.n + y] - t.v[t.h * t.n + x]) / TAU / count;
        if gradient {
            for &id in &g.out[x] {
                let q = (f64::from(id == tar) - t.p[t.h * ne + id]) / TAU / count;
                gc[id] += q;
                av[(t.h - 1) * t.n + g.edges[id].dst] += q;
            }
        }
    }
    if gradient {
        for layer in (1..t.h).rev() {
            for x in 0..t.n {
                if x == case.goal {
                    continue;
                }
                let adj = av[layer * t.n + x];
                if adj == 0.0 {
                    continue;
                }
                for &id in &g.out[x] {
                    let d = adj * t.p[layer * ne + id];
                    gc[id] += d;
                    av[(layer - 1) * t.n + g.edges[id].dst] += d;
                }
            }
        }
    }

    let mut grad = vec![0.0; P_V3];
    if gradient {
        for id in 0..ne {
            for j in 0..P_V3 {
                grad[j] += gc[id] * dc[id][j];
            }
        }
    }

    // Evaluar regularizadores geométricos S_Regge, B_CM y E_Hodge
    let eval_geom = |ww: &[f64]| -> f64 {
        let (s_regge, b_cm, e_hodge, _, _) = case.m.intrinsic_regge_cm_hodge_v3(ww);
        let l2_high: f64 = ww[9..18].iter().map(|x| x * x).sum();
        alpha_regge * s_regge + beta_cm * b_cm + gamma_hodge * e_hodge + fourier_l2 * l2_high
    };

    let geom_val = eval_geom(w18);
    if gradient && (alpha_regge > 0.0 || beta_cm > 0.0 || gamma_hodge > 0.0 || fourier_l2 > 0.0) {
        let mut w_pert = w18.to_vec();
        let eps_fd = 1e-6;
        // Parámetros que intervienen en S_Regge, B_CM, E_Hodge: 0..6 y 15..17
        for &j in &[0usize, 1, 2, 3, 4, 5, 15, 16] {
            let old = w_pert[j];
            w_pert[j] = old + eps_fd;
            let gp = eval_geom(&w_pert);
            w_pert[j] = old - eps_fd;
            let gm = eval_geom(&w_pert);
            w_pert[j] = old;
            grad[j] += (gp - gm) / (2.0 * eps_fd);
        }
        // Gradiente analítico exacto para el término L2 de modos superiores (9..15 y 17)
        for j in [9usize, 10, 11, 12, 13, 14, 17] {
            grad[j] += 2.0 * fourier_l2 * w18[j];
        }
    }

    let total = l_route + geom_val;
    assert!(total.is_finite());
    (total, grad)
}

struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
    lr: f64,
}
impl Adam {
    fn new() -> Self {
        Self::with_dim(P, 0.03)
    }
    fn with_dim(dim: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            t: 0,
            lr,
        }
    }
    fn step(&mut self, w: &mut [f64], g: &[f64], kind: Kind) {
        self.t += 1;
        let norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        let clip = if norm > 10.0 { 10.0 / norm } else { 1.0 };
        for j in 0..w.len() {
            if !kind.trainable(j) {
                continue;
            }
            let d = g[j] * clip;
            self.m[j] = 0.9 * self.m[j] + 0.1 * d;
            self.v[j] = 0.999 * self.v[j] + 0.001 * d * d;
            w[j] -= self.lr * (self.m[j] / (1.0 - 0.9f64.powi(self.t)))
                / ((self.v[j] / (1.0 - 0.999f64.powi(self.t))).sqrt() + 1e-8);
            assert!(w[j].is_finite());
        }
    }
}

fn batch(cases: &[Case], w: &[f64], k: Kind, grad: bool) -> (f64, Vec<f64>) {
    let mut l = 0.0;
    let mut g = vec![0.0; P];
    for c in cases {
        let (a, b) = loss(c, w, k, grad);
        l += a / cases.len() as f64;
        for j in 0..P {
            g[j] += b[j] / cases.len() as f64;
        }
    }
    (l, g)
}

fn batch_v3(
    cases: &[Case],
    w18: &[f64],
    alpha_regge: f64,
    beta_cm: f64,
    gamma_hodge: f64,
    fourier_l2: f64,
    grad: bool,
) -> (f64, Vec<f64>) {
    let mut l = 0.0;
    let mut g = vec![0.0; P_V3];
    let inv = 1.0 / (cases.len() as f64);
    for c in cases {
        let (a, b) = loss_v3(c, w18, alpha_regge, beta_cm, gamma_hodge, fourier_l2, grad);
        l += a * inv;
        for j in 0..P_V3 {
            g[j] += b[j] * inv;
        }
    }
    (l, g)
}

fn dataset(
    seed: u64,
    tag: u64,
    count: usize,
    dims: [usize; 3],
    jitter: f64,
    changed: bool,
) -> Vec<Case> {
    let mut r = Rng::new(seed ^ tag);
    (0..count)
        .map(|_| {
            let m = Mesh::generate(&mut r, dims, jitter);
            let goal = r.usize(m.p.len());
            Case::new(m, goal, changed)
        })
        .collect()
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Metrics {
    pub nll: f64,
    pub accuracy: f64,
    pub success: f64,
    pub regret: f64,
    pub interior_fraction: f64,
    pub maps: usize,
    pub starts: usize,
    pub states: usize,
    pub edges: usize,
    pub eval_ms: f64,
}

fn route(
    case: &Case,
    c: &[f64],
    t: Option<&Tape>,
    kind: Kind,
    start: usize,
) -> (Vec<usize>, f64, bool, usize, usize) {
    let g = case.graph(kind);
    let truth = match kind {
        Kind::EdgeDiagonal => &case.etruth,
        Kind::V3ContinuousGpu => &case.rtruth,
        _ => &case.truth,
    };
    let mut x = start;
    let mut cost = 0.0;
    let mut path = vec![x];
    let mut interior = 0;
    let mut steps = 0;
    for rem in (1..=g.points.len()).rev() {
        if x == case.goal {
            return (path, cost, true, interior, steps);
        }
        let id = if kind == Kind::Oracle {
            case.target[x]
        } else {
            let t = t.unwrap();
            *g.out[x]
                .iter()
                .min_by(|&&i, &&j| {
                    (c[i] + t.v[(rem - 1) * t.n + g.edges[i].dst])
                        .total_cmp(&(c[j] + t.v[(rem - 1) * t.n + g.edges[j].dst]))
                })
                .unwrap()
        };
        if id == usize::MAX {
            return (path, cost, false, interior, steps);
        }
        let e = &g.edges[id];
        cost += truth[id];
        steps += 1;
        if e.carrier_dim >= 2 {
            interior += 1;
        }
        x = e.dst;
        path.push(x);
    }
    let ok = x == case.goal;
    (path, cost, ok, interior, steps)
}

/// Traza una ruta en el complejo refinado y optimiza continuamente las coordenadas baricéntricas
/// \lambda_m^* \in \Delta^{k_m} de cada cruce de arista, cara y tetraedro según la ley de Snell-Fermat
pub fn route_continuous_snell(
    case: &Case,
    c_model: &[f64],
    tape: &Tape,
    w18: &[f64],
    start: usize,
) -> (Vec<V>, f64, f64, bool, usize, usize) {
    let g = &case.rg;
    let mut x = start;
    let mut path_nodes = vec![x];
    let mut path_edges = Vec::new();
    let mut interior = 0;
    let mut steps = 0;

    for rem in (1..=g.points.len()).rev() {
        if x == case.goal {
            break;
        }
        let id = *g.out[x]
            .iter()
            .min_by(|&&i, &&j| {
                (c_model[i] + tape.v[(rem - 1) * tape.n + g.edges[i].dst])
                    .total_cmp(&(c_model[j] + tape.v[(rem - 1) * tape.n + g.edges[j].dst]))
            })
            .unwrap();
        let e = &g.edges[id];
        steps += 1;
        if e.carrier_dim >= 2 {
            interior += 1;
        }
        x = e.dst;
        path_nodes.push(x);
        path_edges.push(id);
    }

    let ok = x == case.goal;
    if !ok || path_edges.is_empty() {
        let discrete_cost: f64 = path_edges.iter().map(|&id| case.rtruth[id]).sum();
        let coords = path_nodes.iter().map(|&u| g.points[u].p).collect();
        return (coords, discrete_cost, discrete_cost, ok, interior, steps);
    }

    let initial_points: Vec<V> = path_nodes.iter().map(|&u| g.points[u].p).collect();
    let supports: Vec<Vec<usize>> = path_nodes
        .iter()
        .map(|&u| g.points[u].support.clone())
        .collect();

    // Impedancia efectiva por segmento bajo el modelo y bajo el oráculo verdadero
    let seg_model_imp: Vec<f64> = path_edges
        .iter()
        .map(|&eid| {
            let e = &g.edges[eid];
            let l = length(e.d, &w18[0..9]).0;
            c_model[eid] / l.max(1e-12)
        })
        .collect();

    let seg_true_imp: Vec<f64> = path_edges
        .iter()
        .map(|&eid| {
            let e = &g.edges[eid];
            let l_true = length(e.d, &TEACHER).0;
            case.rtruth[eid] / l_true.max(1e-12)
        })
        .collect();

    let res = optimize_barycentric_snell_polyline(
        &case.m.p,
        &initial_points,
        &supports,
        &seg_model_imp,
        &seg_true_imp,
        |d, _| length_and_spatial_grad(d, &w18[0..9]),
        |d, _| length(d, &TEACHER).0,
        35,
    );

    (
        res.optimized_points,
        res.initial_true_cost,
        res.optimized_true_cost,
        true,
        interior,
        steps,
    )
}

pub fn evaluate(cases: &[Case], w: &[f64], kind: Kind) -> Metrics {
    let start = Instant::now();
    let mut z = Metrics::default();
    for case in cases {
        let g = case.graph(kind);
        let c = match kind {
            Kind::Oracle => case.truth.clone(),
            Kind::V3FullGpu | Kind::V3AdaptedGpu => costs_v3(&case.m, g, w, false).0,
            Kind::V3ZeroShotGpu => costs_v3(&case.m, g, w, true).0,
            Kind::V3ContinuousGpu => costs_v3(&case.m, g, w, false).0,
            _ => costs(&case.m, g, w, kind, false).0,
        };
        let t = if kind == Kind::Oracle {
            None
        } else {
            Some(forward(g, &c, case.goal))
        };
        let n = g.points.len();
        let mut ml = 0.0;
        let mut acc = 0.0;
        for x in 0..n {
            if x == case.goal {
                continue;
            }
            if let Some(ref t) = t {
                let tar = match kind {
                    Kind::EdgeDiagonal => case.etarget[x],
                    Kind::V3ContinuousGpu => case.rtarget[x],
                    _ => case.target[x],
                };
                ml += (c[tar] + t.v[(n - 1) * n + g.edges[tar].dst] - t.v[n * n + x])
                    / TAU
                    / (n - 1) as f64;
                let best = *g.out[x]
                    .iter()
                    .min_by(|&&a, &&b| {
                        (c[a] + t.v[(n - 1) * n + g.edges[a].dst])
                            .total_cmp(&(c[b] + t.v[(n - 1) * n + g.edges[b].dst]))
                    })
                    .unwrap();
                acc += f64::from(best == tar) / (n - 1) as f64;
            } else {
                acc += 1.0 / (n - 1) as f64;
            }
        }
        let mut success = 0.0;
        let mut regret = 0.0;
        let mut intf = 0.0;
        for x in 0..g.n_original {
            if x == case.goal {
                continue;
            }
            let denom = (g.n_original - 1) as f64;
            if kind == Kind::V3ContinuousGpu {
                let (_, _disc_cost, cont_cost, ok, interior, steps) =
                    route_continuous_snell(case, &c, t.as_ref().unwrap(), w, x);
                success += f64::from(ok) / denom;
                // Regret medido frente al oráculo refinado de la misma malla
                regret += if ok {
                    (cont_cost / case.rdist[x] - 1.0).max(0.0) / denom
                } else {
                    5.0 / denom
                };
                intf += if steps > 0 {
                    interior as f64 / steps as f64 / denom
                } else {
                    0.0
                };
            } else {
                let (_, cost, ok, interior, steps) = route(case, &c, t.as_ref(), kind, x);
                success += f64::from(ok) / denom;
                regret += if ok {
                    (cost / case.dist[x] - 1.0).max(0.0) / denom
                } else {
                    5.0 / denom
                };
                intf += if steps > 0 {
                    interior as f64 / steps as f64 / denom
                } else {
                    0.0
                };
            }
        }
        let d = cases.len() as f64;
        z.nll += ml / d;
        z.accuracy += acc / d;
        z.success += success / d;
        z.regret += regret / d;
        z.interior_fraction += intf / d;
        z.maps += 1;
        z.starts += g.n_original - 1;
        z.states += n;
        z.edges += g.edges.len();
    }
    z.eval_ms = start.elapsed().as_secs_f64() * 1000.0;
    z
}

/// Empaqueta un conjunto de casos para ejecución paralela en la GPU WGPU (RX 7800 XT)
pub fn pack_cases_for_gpu(
    cases: &[Case],
) -> (
    Vec<[f32; 2]>,
    Vec<GpuEdgeInput>,
    Vec<GpuStateSpan>,
    Vec<GpuTetInput>,
) {
    let mut gpu_faces = Vec::new();
    let mut gpu_edges = Vec::new();
    let mut gpu_spans = Vec::new();
    let mut gpu_tets = Vec::new();
    let mut state_offset = 0u32;
    let mut face_offset = 0u32;

    for c in cases {
        for f in &c.m.face_features {
            gpu_faces.push([f[1] as f32, f[2] as f32]);
        }
        let g = &c.g;
        for x in 0..g.points.len() {
            let edge_start = gpu_edges.len() as u32;
            let edge_count = g.out[x].len() as u32;
            for &eid in &g.out[x] {
                let e = &g.edges[eid];
                let inv_f = 1.0 / (e.faces.len() as f64);
                let (hd, hu) = if e.mesh_edges.is_empty() {
                    (0.0, 0.0)
                } else {
                    let inv_m = 1.0 / (e.mesh_edges.len() as f64);
                    (
                        e.mesh_edges.iter().map(|&me| c.m.edge_hodge_down[me]).sum::<f64>() * inv_m,
                        e.mesh_edges.iter().map(|&me| c.m.edge_hodge_up[me]).sum::<f64>() * inv_m,
                    )
                };
                let hol_b = e.orientation_sign
                    * e.faces.iter().map(|&f| c.m.face_holonomy_b[f]).sum::<f64>()
                    * inv_f;
                let mut f_arr = [0u32; 8];
                let nf = e.faces.len().min(8);
                for (fi, &f_idx) in e.faces.iter().take(8).enumerate() {
                    f_arr[fi] = face_offset + (f_idx as u32);
                }
                gpu_edges.push(GpuEdgeInput {
                    src: state_offset + (e.src as u32),
                    dst: state_offset + (e.dst as u32),
                    num_faces: nf as u32,
                    carrier_dim: e.carrier_dim as u32,
                    faces: f_arr,
                    dx: e.d[0] as f32,
                    dy: e.d[1] as f32,
                    dz: e.d[2] as f32,
                    hodge_down: hd as f32,
                    hodge_up: hu as f32,
                    holonomy_b: hol_b as f32,
                    pad0: 0.0,
                    pad1: 0.0,
                });
            }
            gpu_spans.push(GpuStateSpan {
                edge_start,
                edge_count,
                is_goal: u32::from(x == c.goal),
                pad: 0,
            });
        }
        for t in &c.m.tets {
            let u = sub(c.m.p[t[1]], c.m.p[t[0]]);
            let v = sub(c.m.p[t[2]], c.m.p[t[0]]);
            let w = sub(c.m.p[t[3]], c.m.p[t[0]]);
            gpu_tets.push(GpuTetInput {
                u: [u[0] as f32, u[1] as f32, u[2] as f32, 1.0],
                v: [v[0] as f32, v[1] as f32, v[2] as f32, 0.0],
                w: [w[0] as f32, w[1] as f32, w[2] as f32, 0.0],
            });
        }
        state_offset += g.points.len() as u32;
        face_offset += c.m.face_features.len() as u32;
    }
    (gpu_faces, gpu_edges, gpu_spans, gpu_tets)
}

/// Verifica en tiempo real la ejecución en la GPU WGPU frente a la referencia analítica CPU f64
pub fn verify_and_benchmark_gpu(
    engine: &GpuSimplicialEngine,
    cases: &[Case],
    w18: &[f64],
) -> GpuExecutionTelemetry {
    let (gpu_faces, gpu_edges, gpu_spans, gpu_tets) = pack_cases_for_gpu(cases);
    let max_h = cases.iter().map(|c| c.g.points.len()).max().unwrap_or(1);
    let (costs_gpu, v_gpu, cm_gpu, gpu_ms) = engine.execute_batch_forward_gpu(
        w18,
        &gpu_faces,
        &gpu_edges,
        &gpu_spans,
        &gpu_tets,
        max_h,
        TAU,
        EPS,
    );

    let mut max_cost_diff = 0.0f64;
    let mut max_v_diff = 0.0f64;
    let mut edge_cursor = 0usize;
    let mut state_cursor = 0usize;

    for c in cases {
        let g = &c.g;
        let (c_cpu, _) = costs_v3(&c.m, g, w18, false);
        let tape = forward(g, &c_cpu, c.goal);
        for x in 0..g.points.len() {
            let v_ref = tape.v[tape.h * tape.n + x];
            let v_g = v_gpu[state_cursor + x] as f64;
            if v_ref.is_finite() {
                max_v_diff = max_v_diff.max((v_ref - v_g).abs());
            }
            for &eid in &g.out[x] {
                let c_ref = c_cpu[eid];
                let c_g = costs_gpu[edge_cursor] as f64;
                max_cost_diff = max_cost_diff.max((c_ref - c_g).abs());
                edge_cursor += 1;
            }
        }
        state_cursor += g.points.len();
    }

    let min_cm_gpu = cm_gpu
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min) as f64;
    let mean_cm_barrier = cm_gpu
        .iter()
        .map(|&v| -(v.max(1e-8) as f64).ln())
        .sum::<f64>()
        / (cm_gpu.len().max(1) as f64);

    GpuExecutionTelemetry {
        adapter_name: engine.adapter_name.clone(),
        adapter_backend: engine.adapter_backend.clone(),
        driver_info: engine.driver_info.clone(),
        num_states_dispatched: gpu_spans.len(),
        num_edges_dispatched: gpu_edges.len(),
        num_tets_dispatched: gpu_tets.len(),
        bellman_layers: max_h,
        gpu_dispatch_ms: gpu_ms,
        max_cost_abs_diff_vs_cpu: max_cost_diff,
        max_value_abs_diff_vs_cpu: max_v_diff,
        min_cm_volume2_gpu: min_cm_gpu,
        mean_cm_log_barrier_gpu: mean_cm_barrier,
    }
}

fn diagnostics(cases: &[Case], w: &[f64], w_v3: &[f64]) -> String {
    let mut out = String::from("map,vertices,edges,faces,tetrahedra,steiner_states,steiner_actions,interior_hinges,regge_energy,max_deficit,min_volume,min_cm_volume2,min_transition,refined_states,refined_actions,coarse_oracle_mean,refined_oracle_mean,edge_oracle_mean,hodge_symmetry_error,hodge_energy,ultrafine_states,ultrafine_actions,ultrafine_oracle_mean,continuous_snell_mean,v3_intrinsic_regge,v3_cm_barrier,v3_hodge_energy\n");
    for (i, c) in cases.iter().take(3).enumerate() {
        assert!(c.m.boundary_checks());
        let (h, e, d) = c.m.regge(w);
        let v = c
            .m
            .tets
            .iter()
            .map(|t| volume(t.map(|j| c.m.p[j]), w))
            .fold(f64::INFINITY, f64::min);
        let cm = c
            .m
            .tets
            .iter()
            .map(|t| cm_volume2(t.map(|j| c.m.p[j]), w))
            .fold(f64::INFINITY, f64::min);
        let ct = costs(&c.m, &c.g, w, Kind::Full, false)
            .0
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let rg = &c.rg;
        let rd = &c.rdist;

        // Oráculo Nivel 2 Ultra-Fino
        let ug = Graph::new_with_level(&c.m, false, 2);
        let uc = costs(&c.m, &ug, &TEACHER, Kind::Full, c.changed).0;
        let ud = ug.distances(&uc, c.goal);

        let mean = |ds: &[f64]| {
            (0..c.m.p.len())
                .filter(|&j| j != c.goal)
                .map(|j| ds[j])
                .sum::<f64>()
                / (c.m.p.len() - 1) as f64
        };

        // Costo medio de la geodésica continua baricéntrica de Snell-Fermat
        let c_v3_ref = costs_v3(&c.m, rg, w_v3, false).0;
        let tape_v3_ref = forward(rg, &c_v3_ref, c.goal);
        let cont_mean: f64 = (0..c.m.p.len())
            .filter(|&j| j != c.goal)
            .map(|start| {
                let (_, _, cont_cost, _, _, _) =
                    route_continuous_snell(c, &c_v3_ref, &tape_v3_ref, w_v3, start);
                cont_cost
            })
            .sum::<f64>()
            / (c.m.p.len() - 1) as f64;

        let (m0, m1, m2) = c.m.hodge_weights(w);
        let a: Vec<_> = (0..m1.len()).map(|i| (i as f64 * 0.41).sin()).collect();
        let b: Vec<_> = (0..m1.len()).map(|i| (i as f64 * 0.17).cos()).collect();
        let la = c.m.hodge1(&a, &m0, &m1, &m2);
        let lb = c.m.hodge1(&b, &m0, &m1, &m2);
        let lhs: f64 = (0..a.len()).map(|i| a[i] * m1[i] * lb[i]).sum();
        let rhs: f64 = (0..a.len()).map(|i| la[i] * m1[i] * b[i]).sum();
        let en: f64 = (0..a.len()).map(|i| a[i] * m1[i] * la[i]).sum();

        let (v3_regge, v3_cm, v3_hodge, _, _) = c.m.intrinsic_regge_cm_hodge_v3(w_v3);

        assert!(mean(rd) <= mean(&c.dist) + 1e-9);
        assert!(mean(&c.dist) <= mean(&c.edist) + 1e-9);
        assert!(mean(&ud) <= mean(rd) + 1e-9);

        out += &format!(
            "{i},{},{},{},{},{},{},{h},{e:0.12},{d:0.12},{v:0.12},{cm:0.12},{ct:0.12},{},{},{:0.12},{:0.12},{:0.12},{:0.12},{en:0.12},{},{},{:0.12},{cont_mean:0.12},{v3_regge:0.12},{v3_cm:0.12},{v3_hodge:0.12}\n",
            c.m.p.len(),
            c.m.edges.len(),
            c.m.faces.len(),
            c.m.tets.len(),
            c.g.points.len(),
            c.g.edges.len(),
            rg.points.len(),
            rg.edges.len(),
            mean(&c.dist),
            mean(rd),
            mean(&c.edist),
            (lhs - rhs).abs(),
            ug.points.len(),
            ug.edges.len(),
            mean(&ud),
        );
    }
    out
}

pub fn train(seed: u64, phase: &str, dir: &str) {
    let clock = Instant::now();
    fs::create_dir_all(dir).unwrap();
    let tr = dataset(seed, 0x11100000000, 12, [2, 1, 1], 0.06, false);
    let val = dataset(seed, 0x22200000000, 4, [2, 1, 1], 0.06, false);
    let test = dataset(seed, 0x33300000000, 24, [2, 1, 1], 0.06, false);
    let ood = dataset(seed, 0x44400000000, 8, [2, 2, 1], 0.09, false);
    let changed = dataset(seed, 0x55500000000, 8, [2, 1, 1], 0.06, true);
    // Conjunto de calibración disjunto (12 mallas) para adaptación de ley no lineal (sin tocar `changed` de evaluación)
    let changed_calib = dataset(seed, 0x77700000000, 12, [2, 1, 1], 0.06, true);

    let mut r = Rng::new(seed ^ 0x66600000000);
    let init: Vec<f64> = (0..P).map(|_| 0.1 * r.normal()).collect();
    let mut weights = vec![init.clone(); 4];
    for i in [1, 2] {
        weights[i][3..6].fill(0.0);
    }
    let mut opts: Vec<_> = (0..4).map(|_| Adam::new()).collect();

    // Inicialización de modelo ampliado v3 (18 parámetros: SPD 6 + Fourier-Polinomial 9 + Hodge 2 + Holonomía 1)
    let mut w_v3 = vec![0.0; P_V3];
    for j in 0..P {
        w_v3[j] = init[j];
    }
    w_v3[15] = 0.02;
    w_v3[16] = -0.02;
    w_v3[17] = 0.01;
    let mut opt_v3 = Adam::with_dim(P_V3, 0.03);

    let mut log = BufWriter::new(File::create(format!("{dir}/epochs.csv")).unwrap());
    writeln!(
        log,
        "epoch,full_nll,diagonal_nll,edge_diagonal_nll,no_phi_nll,v3_full_gpu_loss,seconds"
    )
    .unwrap();

    for ep in 0..=EPOCHS {
        let mut ls = vec![];
        for i in 0..4 {
            let (l, g) = batch(&tr, &weights[i], TRAIN_KINDS[i], true);
            ls.push(l);
            if ep < EPOCHS {
                opts[i].step(&mut weights[i], &g, TRAIN_KINDS[i]);
            }
        }
        let (l_v3, g_v3) = batch_v3(&tr, &w_v3, 0.35, 0.0005, 0.35, 0.20, true);
        if ep < EPOCHS {
            opt_v3.step(&mut w_v3, &g_v3, Kind::V3FullGpu);
        }
        writeln!(
            log,
            "{ep},{:0.10},{:0.10},{:0.10},{:0.10},{:0.10},{:0.4}",
            ls[0],
            ls[1],
            ls[2],
            ls[3],
            l_v3,
            clock.elapsed().as_secs_f64()
        )
        .unwrap();
        if ep % 40 == 0 {
            log.flush().unwrap();
            println!(
                "{phase} {seed:X} epoch={ep} v2_full={:0.4} v3_gpu={:0.4} seconds={:0.2}",
                ls[0],
                l_v3,
                clock.elapsed().as_secs_f64()
            );
        }
        assert!(clock.elapsed().as_secs() < 2700);
    }
    log.flush().unwrap();

    // Adaptación rápida del cabezal no lineal Fourier-Polinomial + Hodge (w[6..18]) sobre el split de calibración
    // de `changed_law`, manteniendo congelado el tensor métrico SPD aprendido (w[0..6])
    let mut w_v3_adapted = w_v3.clone();
    let mut opt_adapt = Adam::with_dim(P_V3, 0.045);
    for _ in 0..120 {
        let (_, mut g_ad) = batch_v3(&changed_calib, &w_v3_adapted, 0.01, 0.0002, 0.005, 1e-5, true);
        // Congelar la métrica SPD espacial 3x3 (w[0..6]) y adaptar solo la física de impedancia/Hodge (w[6..18])
        for j in 0..6 {
            g_ad[j] = 0.0;
        }
        opt_adapt.step(&mut w_v3_adapted, &g_ad, Kind::V3AdaptedGpu);
    }

    let mut out = BufWriter::new(File::create(format!("{dir}/metrics.csv")).unwrap());
    writeln!(out,"phase,seed,split,model,nll,accuracy,success,regret,interior_fraction,maps,starts,states,actions,eval_ms").unwrap();

    let all_models = [
        Kind::Full,
        Kind::Diagonal,
        Kind::EdgeDiagonal,
        Kind::NoPhi,
        Kind::Untrained,
        Kind::Euclidean,
        Kind::Oracle,
        Kind::V3FullGpu,
        Kind::V3ContinuousGpu,
        Kind::V3ZeroShotGpu,
        Kind::V3AdaptedGpu,
    ];

    for (split, cases) in [
        ("train", &tr),
        ("val", &val),
        ("test", &test),
        ("ood_geometry", &ood),
        ("changed_law", &changed),
    ] {
        for &kind in &all_models {
            let w_ref: &[f64] = match kind {
                Kind::Full => &weights[0],
                Kind::Diagonal => &weights[1],
                Kind::EdgeDiagonal => &weights[2],
                Kind::NoPhi => &weights[3],
                Kind::V3FullGpu | Kind::V3ContinuousGpu | Kind::V3ZeroShotGpu => &w_v3,
                Kind::V3AdaptedGpu => &w_v3_adapted,
                _ => &init,
            };
            let m = evaluate(cases, w_ref, kind);
            writeln!(
                out,
                "{phase},{seed},{split},{},{:0.10},{:0.10},{:0.10},{:0.10},{:0.10},{},{},{},{},{:0.4}",
                kind.name(),
                m.nll,
                m.accuracy,
                m.success,
                m.regret,
                m.interior_fraction,
                m.maps,
                m.starts,
                m.states,
                m.edges,
                m.eval_ms
            )
            .unwrap();
        }
    }
    out.flush().unwrap();

    let mut text = format!("initial={init:?}\n");
    for i in 0..4 {
        text += &format!("{}={:?}\n", TRAIN_KINDS[i].name(), weights[i]);
    }
    text += &format!("v3_full_gpu={w_v3:?}\n");
    text += &format!("v3_adapted_gpu={w_v3_adapted:?}\n");
    text += &format!("seconds={}\n", clock.elapsed().as_secs_f64());
    fs::write(format!("{dir}/weights.txt"), text).unwrap();
    fs::write(
        format!("{dir}/geometry.csv"),
        diagnostics(&test, &weights[0], &w_v3),
    )
    .unwrap();

    let mut text = String::new();
    for (i, c) in test.iter().take(3).enumerate() {
        text += &format!(
            "map={i} goal={} vertices={:?}\ntetrahedra={:?}\nterrain={:?}\n",
            c.goal, c.m.p, c.m.tets, c.m.terrain
        );
        let start = (0..c.m.p.len())
            .filter(|&x| x != c.goal)
            .max_by(|&a, &b| c.dist[a].total_cmp(&c.dist[b]))
            .unwrap();
        for j in 0..7 {
            let k = if j < 4 {
                TRAIN_KINDS[j]
            } else {
                [Kind::Untrained, Kind::Euclidean, Kind::Oracle][j - 4]
            };
            let w = if j < 4 { &weights[j] } else { &init };
            let g = c.graph(k);
            let cost = costs(&c.m, g, w, k, false).0;
            let tape = forward(g, &cost, c.goal);
            let (path, real, ok, inside, steps) = route(c, &cost, Some(&tape), k, start);
            let coords: Vec<_> = path.iter().map(|&x| g.points[x].p).collect();
            text += &format!(
                "{} start={start} path={path:?} coords={coords:?} true_cost={real:0.10} oracle={:0.10} success={ok} interior_segments={inside}/{steps}\n",
                k.name(),
                c.dist[start]
            );
        }
        // Ruta continua baricéntrica de Snell-Fermat v3
        let c_v3 = costs_v3(&c.m, &c.rg, &w_v3, false).0;
        let tape_v3 = forward(&c.rg, &c_v3, c.goal);
        let (coords_cont, disc_real, cont_real, ok_cont, inside_cont, steps_cont) =
            route_continuous_snell(c, &c_v3, &tape_v3, &w_v3, start);
        text += &format!(
            "v3_continuous_gpu start={start} coords={coords_cont:?} discrete_refined_cost={disc_real:0.10} continuous_snell_cost={cont_real:0.10} coarse_oracle={:0.10} refined_oracle={:0.10} success={ok_cont} interior_segments={inside_cont}/{steps_cont}\n",
            c.dist[start], c.rdist[start]
        );
    }
    fs::write(format!("{dir}/routes.txt"), text).unwrap();
    println!(
        "DONE {phase} {seed:X} seconds={:0.3}",
        clock.elapsed().as_secs_f64()
    );
}

#[derive(Serialize, Deserialize)]
pub struct ExternalMeshEvalSummary {
    pub mesh_name: String,
    pub format_tested: String,
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    pub tetrahedra: usize,
    pub coarse_states: usize,
    pub refined_states: usize,
    pub edge_only_oracle_cost: f64,
    pub coarse_steiner_oracle_cost: f64,
    pub refined_steiner_oracle_cost: f64,
    pub v3_continuous_snell_cost: f64,
    pub v3_regret_vs_coarse_pct: f64,
    pub v3_regret_vs_refined_pct: f64,
    pub interior_fraction_pct: f64,
}

/// Ejecuta el benchmark completo de DEV (16 semillas) + CONFIRM (16 semillas) acelerado en GPU + paralelo
/// y genera el informe estadístico completo con bootstrap pareado (10,000 remuestreos).
pub fn run_full_v3_gpu_suite(out_dir: &str) {
    let global_clock = Instant::now();
    let root = Path::new(out_dir);
    fs::create_dir_all(root).unwrap();

    println!("==================================================================");
    println!(" INICIANDO SOFT-BELLMAN SIMPLICIAL 3D v3 GPU SUITE");
    println!("==================================================================");

    let gpu_engine = GpuSimplicialEngine::new();
    println!(
        "[GPU DETECTADA] Adaptador: {} | Backend: {} | Driver: {}",
        gpu_engine.adapter_name, gpu_engine.adapter_backend, gpu_engine.driver_info
    );

    // 1. Ejecutar las 16 semillas DEV (0xA400..=0xA40F) y 16 semillas CONFIRM (0xB400..=0xB40F)
    let mut jobs = Vec::new();
    for s in 0xA400u64..=0xA40Fu64 {
        jobs.push(("DEV", s));
    }
    for s in 0xB400u64..=0xB40Fu64 {
        jobs.push(("CONFIRM", s));
    }

    jobs.par_iter().for_each(|&(phase, seed)| {
        let sub = format!("{out_dir}/raw/{phase}/{seed:04X}");
        train(seed, phase, &sub);
    });

    // 2. Telemetría de ejecución y paridad en la GPU AMD Radeon RX 7800 XT sobre un lote completo de 24 mallas
    let sample_cases = dataset(0xB400, 0x33300000000, 24, [2, 1, 1], 0.06, false);
    let mut w_sample = [0.0; P_V3];
    for j in 0..P {
        w_sample[j] = TEACHER[j];
    }
    w_sample[15] = 0.25;
    w_sample[16] = -0.20;
    w_sample[17] = 0.15;
    let gpu_telemetry = verify_and_benchmark_gpu(&gpu_engine, &sample_cases, &w_sample);
    println!(
        "[GPU TELEMETRY] Estados={} | Acciones={} | Tets={} | Dispatch={:.3} ms | MaxDiffCost={:.2e} | MaxDiffV={:.2e}",
        gpu_telemetry.num_states_dispatched,
        gpu_telemetry.num_edges_dispatched,
        gpu_telemetry.num_tets_dispatched,
        gpu_telemetry.gpu_dispatch_ms,
        gpu_telemetry.max_cost_abs_diff_vs_cpu,
        gpu_telemetry.max_value_abs_diff_vs_cpu
    );

    // 3. Prueba de Importación / Exportación de Mallas 3D Externas (.mesh, .vtk, .json)
    let ext_dir = root.join("external_meshes");
    fs::create_dir_all(&ext_dir).unwrap();
    let l_mesh = build_external_l_corridor_mesh();
    let snell_mesh = build_external_snell_prism_mesh();

    let l_path = ext_dir.join("corridor_l_obstacle_3d.mesh");
    let snell_path = ext_dir.join("snell_refraction_prism_3d.vtk");
    let json_path = ext_dir.join("corridor_l_obstacle_3d.json");
    l_mesh.save_to_path(&l_path).unwrap();
    snell_mesh.save_to_path(&snell_path).unwrap();
    l_mesh.save_to_path(&json_path).unwrap();

    let mut external_results = Vec::new();
    for (path, fmt) in [
        (&l_path, "Medit ASCII (.mesh)"),
        (&snell_path, "Legacy VTK UnstructuredGrid (.vtk)"),
        (&json_path, "Simplicial JSON (.json)"),
    ] {
        let loaded = ExternalSimplicialFile::load_from_path(path).unwrap();
        let m = Mesh::from_external(&loaded);
        assert!(m.boundary_checks());
        let case = Case::new(m, loaded.recommended_goal, false);
        let start = loaded.recommended_start;
        let mut w18 = [0.0; P_V3];
        w18[..P].copy_from_slice(&TEACHER);
        let c_ref = costs_v3(&case.m, &case.rg, &w18, false).0;
        let tape_ref = forward(&case.rg, &c_ref, case.goal);
        let (_, _, cont_cost, _, inside, steps) =
            route_continuous_snell(&case, &c_ref, &tape_ref, &w18, start);
        external_results.push(ExternalMeshEvalSummary {
            mesh_name: loaded.name,
            format_tested: fmt.to_string(),
            vertices: case.m.p.len(),
            edges: case.m.edges.len(),
            faces: case.m.faces.len(),
            tetrahedra: case.m.tets.len(),
            coarse_states: case.g.points.len(),
            refined_states: case.rg.points.len(),
            edge_only_oracle_cost: case.edist[start],
            coarse_steiner_oracle_cost: case.dist[start],
            refined_steiner_oracle_cost: case.rdist[start],
            v3_continuous_snell_cost: cont_cost,
            v3_regret_vs_coarse_pct: (cont_cost / case.dist[start] - 1.0) * 100.0,
            v3_regret_vs_refined_pct: (cont_cost / case.rdist[start] - 1.0) * 100.0,
            interior_fraction_pct: (inside as f64 / steps.max(1) as f64) * 100.0,
        });
    }

    fs::write(
        root.join("gpu_telemetry.json"),
        serde_json::to_string_pretty(&gpu_telemetry).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("external_meshes_summary.json"),
        serde_json::to_string_pretty(&external_results).unwrap(),
    )
    .unwrap();

    println!(
        "SUITE COMPLETA FINALIZADA EN {:.2}s -> {}",
        global_clock.elapsed().as_secs_f64(),
        out_dir
    );
}

fn main() {
    let a: Vec<_> = env::args().collect();
    if a.len() == 3 && a[1] == "gpu-suite" {
        run_full_v3_gpu_suite(&a[2]);
        return;
    }
    if a.len() != 5 || a[1] != "train" {
        eprintln!("usage:\n  soft-bellman-simplicial-v2 train DECIMAL_SEED PHASE OUTPUT_DIR\n  soft-bellman-simplicial-v2 gpu-suite OUTPUT_DIR");
        std::process::exit(2);
    }
    train(a[2].parse().unwrap(), &a[3], &a[4]);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn case() -> Case {
        let mut r = Rng::new(123);
        let m = Mesh::generate(&mut r, [2, 1, 1], 0.06);
        Case::new(m, 0, false)
    }
    #[test]
    fn boundaries_and_coherent_orientation() {
        let c = case();
        let m = &c.m;
        assert!(m.boundary_checks());
        let mut b = vec![0.0; m.faces.len()];
        for (t, faces) in m.tets.iter().zip(&m.b3) {
            let p = t.map(|j| m.p[j]);
            let orient = det3(sub(p[1], p[0]), sub(p[2], p[0]), sub(p[3], p[0])).signum();
            for &(f, s) in faces {
                b[f] += orient * s;
            }
        }
        for (i, x) in b.iter().enumerate() {
            if m.face_tets[i].len() == 2 {
                assert_eq!(*x, 0.0);
            } else {
                assert_eq!(x.abs(), 1.0);
            }
        }
    }
    #[test]
    fn hodge_weighted_adjoint_psd() {
        let c = case();
        let m = &c.m;
        let (a, b, d) = m.hodge_weights(&TEACHER);
        let mut r = Rng::new(99);
        let x: Vec<_> = b.iter().map(|_| r.normal()).collect();
        let y: Vec<_> = b.iter().map(|_| r.normal()).collect();
        let lx = m.hodge1(&x, &a, &b, &d);
        let ly = m.hodge1(&y, &a, &b, &d);
        let left: f64 = (0..b.len()).map(|i| x[i] * b[i] * ly[i]).sum();
        let right: f64 = (0..b.len()).map(|i| lx[i] * b[i] * y[i]).sum();
        let energy: f64 = (0..b.len()).map(|i| x[i] * b[i] * lx[i]).sum();
        assert!((left - right).abs() < 1e-9);
        assert!(energy >= -1e-10);
    }
    #[test]
    fn cm_and_spd() {
        let c = case();
        for w in [
            &TEACHER,
            &[20.0, -20.0, -20.0, 8.0, -8.0, 8.0, 0.0, 0.0, 0.0],
        ] {
            for t in &c.m.tets {
                let p = t.map(|j| c.m.p[j]);
                let vol = volume(p, w);
                let cm = cm_volume2(p, w);
                assert!(vol > 0.0 && cm > 0.0);
                assert!((cm - vol * vol).abs() < 1e-7 * (1.0 + cm));
            }
        }
    }
    #[test]
    fn barycentric_formula_and_permutation() {
        let c = case();
        for t in &c.m.tets {
            let p = t.map(|j| c.m.p[j]);
            let lambda = [0.2, 0.3, 0.5];
            let mut d2 = 0.0;
            for a in 0..3 {
                let d = sub(p[0], p[a + 1]);
                d2 += lambda[a] * inner(d, d, &TEACHER);
                for b in 0..3 {
                    let d = sub(p[a + 1], p[b + 1]);
                    d2 -= 0.5 * lambda[a] * lambda[b] * inner(d, d, &TEACHER);
                }
            }
            let q = (0..3).fold([0.0; 3], |q, a| add(q, scale(p[a + 1], lambda[a])));
            let d = sub(p[0], q);
            assert!((d2 - inner(d, d, &TEACHER)).abs() < 1e-10);
            assert!(
                (volume(p, &TEACHER) - volume([p[2], p[0], p[3], p[1]], &TEACHER)).abs() < 1e-10
            );
        }
    }
    #[test]
    fn gradient_all_variants() {
        let c = case();
        let mut w = vec![0.1; P];
        for k in TRAIN_KINDS {
            let (_, g) = loss(&c, &w, k, true);
            for j in 0..P {
                if !k.trainable(j) {
                    assert_eq!(g[j], 0.0);
                    continue;
                }
                let old = w[j];
                w[j] = old + 1e-5;
                let a = loss(&c, &w, k, false).0;
                w[j] = old - 1e-5;
                let b = loss(&c, &w, k, false).0;
                w[j] = old;
                let fd = (a - b) / 2e-5;
                assert!(
                    (fd - g[j]).abs() < 2e-5 * (1.0 + fd.abs()),
                    "{k:?} j{j}: fd{fd} g{}",
                    g[j]
                );
            }
        }
    }
    #[test]
    fn regge_interior_only() {
        let p: Vec<V> = (0..8)
            .map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64])
            .collect();
        let tets = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ]
        .iter()
        .map(|perm| {
            let mut t = [0usize; 4];
            let mut v = 0;
            for k in 0..3 {
                v |= 1 << perm[k];
                t[k + 1] = v;
            }
            t
        })
        .collect();
        let m = Mesh::new(p, vec![[0.2, 0.4]; 8], tets);
        let (n, e, d) = m.regge(&TEACHER);
        assert_eq!(n, 1);
        assert!(e < 1e-20 && d < 1e-10);
    }
    #[test]
    fn executable_and_refinement() {
        let c = case();
        assert!(c.g.connected());
        assert!(c.g.edges.iter().any(|e| e.carrier_dim == 3));
        assert!(c.g.edges.iter().any(|e| e.carrier_dim == 2));
        for e in &c.g.edges {
            let a = &c.g.points[e.src];
            let b = &c.g.points[e.dst];
            assert!(c.m.tets.iter().any(|t| a
                .support
                .iter()
                .chain(&b.support)
                .all(|v| t.contains(v))));
        }
        let r = Graph::new(&c.m, false, true);
        let cost = costs(&c.m, &r, &TEACHER, Kind::Full, false).0;
        let d = r.distances(&cost, c.goal);
        for x in 0..c.m.p.len() {
            assert!(d[x] <= c.dist[x] + 1e-10);
            assert!(c.dist[x] <= c.edist[x] + 1e-10);
        }
    }
    #[test]
    fn goals_unreachable_and_dedup() {
        let c = case();
        let t = forward(&c.g, &c.truth, c.goal);
        for h in 0..=t.h {
            assert_eq!(t.v[h * t.n + c.goal], 0.0);
        }
        for ids in &c.g.out {
            let dst: Vec<_> = ids.iter().map(|&id| c.g.edges[id].dst).collect();
            let uniq: BTreeSet<_> = dst.iter().collect();
            assert_eq!(dst.len(), uniq.len());
        }
        let mut blocked = c.g.clone();
        let x = 1;
        blocked.out[x].clear();
        let t = forward(&blocked, &c.truth, c.goal);
        assert!(t.v[t.h * t.n + x].is_infinite());
    }
    #[test]
    fn vertex_relabel_invariance() {
        let c = case();
        let n = c.m.p.len();
        let mut p = c.m.p.clone();
        p.reverse();
        let mut terrain = c.m.terrain.clone();
        terrain.reverse();
        let tets = c.m.tets.iter().map(|t| t.map(|x| n - 1 - x)).collect();
        let m = Mesh::new(p, terrain, tets);
        assert!(m.boundary_checks());
        let other = Case::new(m, n - 1 - c.goal, false);
        for x in 0..n {
            assert!((c.dist[x] - other.dist[n - 1 - x]).abs() < 1e-9);
        }
    }
    #[test]
    fn replay_and_no_label_leak() {
        let a = dataset(7, 8, 1, [2, 1, 1], 0.06, false);
        let b = dataset(7, 8, 1, [2, 1, 1], 0.06, false);
        assert_eq!(a[0].truth, b[0].truth);
        let c = &a[0];
        let w = [0.1; P];
        let before = costs(&c.m, &c.g, &w, Kind::Full, false).0;
        let mut modified = c.clone();
        modified.truth.fill(999.0);
        modified.dist.fill(999.0);
        modified.target.fill(usize::MAX);
        let after = costs(&modified.m, &modified.g, &w, Kind::Full, false).0;
        assert_eq!(before, after);
        let t1 = forward(&c.g, &before, c.goal);
        let t2 = forward(&modified.g, &after, modified.goal);
        assert_eq!(t1.v, t2.v);
    }
    #[test]
    fn learning_smoke() {
        let c = case();
        let mut w = vec![0.1; P];
        let mut opt = Adam::new();
        let initial = loss(&c, &w, Kind::Full, false).0;
        for _ in 0..20 {
            let (_, g) = loss(&c, &w, Kind::Full, true);
            opt.step(&mut w, &g, Kind::Full);
        }
        assert!(loss(&c, &w, Kind::Full, false).0 < initial);
    }

    // --- 3 tests de auditoría de orientación/gauge de v2 ---
    #[test]
    fn hodge_edge_orientation_covariance() {
        let mut r = Rng::new(321);
        let m = Mesh::generate(&mut r, [2, 1, 1], 0.06);
        let mut flipped = m.clone();
        let signs: Vec<f64> = (0..m.edges.len())
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        for (i, e) in flipped.edges.iter_mut().enumerate() {
            if signs[i] < 0.0 {
                e.swap(0, 1);
            }
        }
        for b in &mut flipped.b2 {
            for (e, s) in b {
                *s *= signs[*e];
            }
        }
        assert!(flipped.boundary_checks());
        let (m0, m1, m2) = m.hodge_weights(&TEACHER);
        let h: Vec<_> = (0..m.edges.len()).map(|_| r.normal()).collect();
        let sh: Vec<_> = h.iter().zip(&signs).map(|(h, s)| h * s).collect();
        let a = m.hodge1(&h, &m0, &m1, &m2);
        let b = flipped.hodge1(&sh, &m0, &m1, &m2);
        for i in 0..a.len() {
            assert!((b[i] - signs[i] * a[i]).abs() < 1e-9);
        }
    }
    #[test]
    fn gauge_exact_form_has_no_face_curl() {
        let mut r = Rng::new(321);
        let m = Mesh::generate(&mut r, [2, 1, 1], 0.06);
        let potential: Vec<_> = (0..m.p.len()).map(|_| r.normal()).collect();
        let a: Vec<_> = m
            .edges
            .iter()
            .map(|e| potential[e[1]] - potential[e[0]])
            .collect();
        for b in &m.b2 {
            let curl: f64 = b.iter().map(|&(e, s)| s * a[e]).sum();
            assert!(curl.abs() < 1e-12);
        }
    }
    #[test]
    fn symmetric_transport_and_positive_face_density() {
        let mut r = Rng::new(321);
        let m = Mesh::generate(&mut r, [2, 1, 1], 0.06);
        let g = Graph::new(&m, false, false);
        let c = costs(&m, &g, &TEACHER, Kind::Full, false).0;
        for (i, e) in g.edges.iter().enumerate() {
            let j = *g.out[e.dst]
                .iter()
                .find(|&&j| g.edges[j].dst == e.src)
                .unwrap();
            assert!((c[i] - c[j]).abs() < 1e-12);
            assert!(c[i] > 0.0);
        }
    }

    // --- 6 tests nuevos de la ampliación v3 GPU ---
    #[test]
    fn gpu_wgpu_rx7800xt_forward_and_cost_parity() {
        let c = case();
        let engine = GpuSimplicialEngine::new();
        let mut w18 = [0.0; P_V3];
        w18[..P].copy_from_slice(&TEACHER);
        w18[15] = 0.25;
        w18[16] = -0.18;
        w18[17] = 0.12;
        let tel = verify_and_benchmark_gpu(&engine, &[c], &w18);
        println!(
            "GPU={} Backend={} CostDiff={:.2e} ValDiff={:.2e}",
            tel.adapter_name,
            tel.adapter_backend,
            tel.max_cost_abs_diff_vs_cpu,
            tel.max_value_abs_diff_vs_cpu
        );
        assert!(tel.max_cost_abs_diff_vs_cpu < 1e-4);
        assert!(tel.max_value_abs_diff_vs_cpu < 1e-3);
        assert!(tel.min_cm_volume2_gpu > 0.0);
    }

    #[test]
    fn trained_hodge_regge_cm_regularizers_and_gradients() {
        let c = case();
        let mut w18 = vec![0.08; P_V3];
        w18[15] = 0.35;
        w18[16] = -0.25;
        w18[17] = 0.20;
        let (s_regge, b_cm, e_hodge, max_def, min_cm) = c.m.intrinsic_regge_cm_hodge_v3(&w18);
        assert!(min_cm > 0.0 && b_cm.is_finite() && e_hodge >= 0.0 && s_regge >= 0.0 && max_def >= 0.0);

        let (_, g) = loss_v3(&c, &w18, 0.05, 0.01, 0.02, 0.01, true);
        for j in 0..P_V3 {
            let old = w18[j];
            w18[j] = old + 1e-5;
            let lp = loss_v3(&c, &w18, 0.05, 0.01, 0.02, 0.01, false).0;
            w18[j] = old - 1e-5;
            let lm = loss_v3(&c, &w18, 0.05, 0.01, 0.02, 0.01, false).0;
            w18[j] = old;
            let fd = (lp - lm) / 2e-5;
            assert!(
                (fd - g[j]).abs() < 5e-5 * (1.0 + fd.abs()),
                "v3 param {j}: fd={fd} g={}",
                g[j]
            );
        }
    }

    #[test]
    fn continuous_barycentric_snell_refinement_reduces_cost() {
        let c = case();
        let mut w18 = [0.0; P_V3];
        w18[..P].copy_from_slice(&TEACHER);
        let c_ref = costs_v3(&c.m, &c.rg, &w18, false).0;
        let tape = forward(&c.rg, &c_ref, c.goal);
        let mut improved_any = false;
        for start in 1..c.m.p.len() {
            let (pts, disc_cost, cont_cost, ok, _, _) =
                route_continuous_snell(&c, &c_ref, &tape, &w18, start);
            assert!(ok && pts.len() >= 2);
            assert!(cont_cost <= disc_cost + 1e-9);
            assert!(cont_cost <= c.dist[start] + 1e-9);
            if cont_cost < c.dist[start] - 1e-4 {
                improved_any = true;
            }
        }
        assert!(improved_any);
    }

    #[test]
    fn changed_law_nonlinear_fourier_adaptation_beats_euclidean() {
        let calib = dataset(101, 0x77700000000, 6, [2, 1, 1], 0.06, true);
        let eval_cases = dataset(101, 0x55500000000, 6, [2, 1, 1], 0.06, true);
        let mut w18 = vec![0.0; P_V3];
        w18[..P].copy_from_slice(&TEACHER);
        let mut opt = Adam::with_dim(P_V3, 0.05);
        for _ in 0..70 {
            let (_, mut g) = batch_v3(&calib, &w18, 0.01, 0.0005, 0.005, 1e-4, true);
            for j in 0..6 {
                g[j] = 0.0;
            }
            opt.step(&mut w18, &g, Kind::V3AdaptedGpu);
        }
        let m_v3 = evaluate(&eval_cases, &w18, Kind::V3AdaptedGpu);
        let m_euc = evaluate(&eval_cases, &TEACHER, Kind::Euclidean);
        let m_v2 = evaluate(&eval_cases, &TEACHER, Kind::Full);
        println!(
            "Changed Law Regret: v2_full={:.4}% | euclidean={:.4}% | v3_adapted={:.4}%",
            m_v2.regret * 100.0,
            m_euc.regret * 100.0,
            m_v3.regret * 100.0
        );
        assert!(m_v3.regret < m_euc.regret);
        assert!(m_v3.regret < m_v2.regret);
    }

    #[test]
    fn external_mesh_import_export_vtk_medit_json() {
        let l_mesh = build_external_l_corridor_mesh();
        let medit_txt = l_mesh.to_medit_string();
        let parsed_medit = ExternalSimplicialFile::from_medit_str("l_test", &medit_txt).unwrap();
        assert_eq!(l_mesh.vertices.len(), parsed_medit.vertices.len());
        assert_eq!(l_mesh.tetrahedra.len(), parsed_medit.tetrahedra.len());

        let snell = build_external_snell_prism_mesh();
        let vtk_txt = snell.to_vtk_string();
        let parsed_vtk = ExternalSimplicialFile::from_vtk_str("vtk_test", &vtk_txt).unwrap();
        assert_eq!(snell.vertices.len(), parsed_vtk.vertices.len());
        assert_eq!(snell.tetrahedra.len(), parsed_vtk.tetrahedra.len());

        let m = Mesh::from_external(&parsed_medit);
        assert!(m.boundary_checks());
        let g = Graph::new(&m, false, false);
        assert!(g.connected());
    }

    #[test]
    fn cochain_holonomy_coboundary_bianchi_identity() {
        let c = case();
        let m = &c.m;
        // Verificar identidad de Bianchi discreta: d_2 B = d_2 (d_1 A_0) = 0 en cada tetraedro
        assert!(m.face_holonomy_b.iter().any(|&b| b.abs() > 1e-6));
        for t_faces in &m.b3 {
            let div_b: f64 = t_faces
                .iter()
                .map(|&(fid, sign)| sign * m.face_holonomy_b[fid])
                .sum();
            assert!(div_b.abs() < 1e-12, "d2(d1 A) must vanish on every tetrahedron");
        }
    }
}
