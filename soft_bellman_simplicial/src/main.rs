//! Executable Steiner approximation on an oriented tetrahedral 3D complex.
//! Rust-only generation, geometry, analytic cost derivatives, reverse Bellman and Adam.
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    time::Instant,
};
const P: usize = 9;
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
#[derive(Clone)]
struct Mesh {
    p: Vec<V>,
    terrain: Vec<[f64; 2]>,
    tets: Vec<[usize; 4]>,
    faces: Vec<[usize; 3]>,
    edges: Vec<[usize; 2]>,
    b2: Vec<Vec<(usize, f64)>>,
    b3: Vec<Vec<(usize, f64)>>,
    face_tets: Vec<Vec<usize>>,
    face_features: Vec<[f64; 3]>,
}
impl Mesh {
    fn new(p: Vec<V>, terrain: Vec<[f64; 2]>, mut tets: Vec<[usize; 4]>) -> Self {
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
        let face_features = faces
            .iter()
            .map(|f| {
                [
                    1.0,
                    f.iter().map(|&x| terrain[x][0]).sum::<f64>() / 3.0,
                    f.iter().map(|&x| terrain[x][1]).sum::<f64>() / 3.0,
                ]
            })
            .collect();
        Self {
            p,
            terrain,
            tets,
            faces,
            edges,
            b2,
            b3,
            face_tets,
            face_features,
        }
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
    fn boundary_checks(&self) -> bool {
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
    fn hodge_weights(&self, w: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
    fn hodge1(&self, h: &[f64], m0: &[f64], m1: &[f64], m2: &[f64]) -> Vec<f64> {
        let mut div = vec![0.0; self.p.len()];
        for (i, e) in self.edges.iter().enumerate() {
            div[e[0]] -= m1[i] * h[i];
            div[e[1]] += m1[i] * h[i];
        }
        for i in 0..div.len() {
            div[i] /= m0[i];
        }
        let curl: Vec<_> = self
            .b2
            .iter()
            .enumerate()
            .map(|(f, b)| m2[f] * b.iter().map(|&(e, s)| s * h[e]).sum::<f64>())
            .collect();
        let mut out: Vec<_> = self.edges.iter().map(|e| div[e[1]] - div[e[0]]).collect();
        for (f, b) in self.b2.iter().enumerate() {
            for &(e, s) in b {
                out[e] += s * curl[f] / m1[e];
            }
        }
        out
    }
    fn regge(&self, w: &[f64]) -> (usize, f64, f64) {
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
}
#[derive(Clone)]
struct Point {
    p: V,
    support: Vec<usize>,
}
#[derive(Clone)]
struct Edge {
    src: usize,
    dst: usize,
    d: V,
    faces: Vec<usize>,
    carrier_dim: usize,
}
#[derive(Clone)]
struct Graph {
    points: Vec<Point>,
    edges: Vec<Edge>,
    out: Vec<Vec<usize>>,
    n_original: usize,
}
impl Graph {
    fn new(m: &Mesh, edge_only: bool, refine: bool) -> Self {
        let mut points: Vec<_> =
            m.p.iter()
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
            if refine {
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
            for (src, dst) in [(a, b), (b, a)] {
                out[src].push(edges.len());
                edges.push(Edge {
                    src,
                    dst,
                    d: sub(points[dst].p, points[src].p),
                    faces: faces.clone(),
                    carrier_dim,
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
    fn connected(&self) -> bool {
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
    fn distances(&self, c: &[f64], goal: usize) -> Vec<f64> {
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
    fn targets(&self, c: &[f64], d: &[f64], goal: usize) -> Vec<usize> {
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
enum Kind {
    Full,
    Diagonal,
    EdgeDiagonal,
    NoPhi,
    Untrained,
    Euclidean,
    Oracle,
}
impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Diagonal => "diagonal",
            Self::EdgeDiagonal => "edge_diagonal",
            Self::NoPhi => "no_phi",
            Self::Untrained => "untrained",
            Self::Euclidean => "euclidean",
            Self::Oracle => "oracle",
        }
    }
    fn trainable(self, j: usize) -> bool {
        match self {
            Self::Diagonal | Self::EdgeDiagonal => j < 3 || j >= 6,
            Self::NoPhi => j < 6,
            Self::Full => true,
            _ => false,
        }
    }
}
const TRAIN_KINDS: [Kind; 4] = [Kind::Full, Kind::Diagonal, Kind::EdgeDiagonal, Kind::NoPhi];
fn costs(m: &Mesh, g: &Graph, w: &[f64], kind: Kind, changed: bool) -> (Vec<f64>, Vec<[f64; P]>) {
    let mut ww = w.to_vec();
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
const TEACHER: [f64; P] = [0.15, -0.15, 0.05, 1.2, -0.95, 0.8, -0.7, 2.0, 1.2];
#[derive(Clone)]
struct Case {
    m: Mesh,
    g: Graph,
    eg: Graph,
    goal: usize,
    truth: Vec<f64>,
    etruth: Vec<f64>,
    dist: Vec<f64>,
    edist: Vec<f64>,
    target: Vec<usize>,
    etarget: Vec<usize>,
    changed: bool,
}
impl Case {
    fn new(m: Mesh, goal: usize, changed: bool) -> Self {
        let g = Graph::new(&m, false, false);
        let eg = Graph::new(&m, true, false);
        let truth = costs(&m, &g, &TEACHER, Kind::Full, changed).0;
        let etruth = costs(&m, &eg, &TEACHER, Kind::Full, changed).0;
        let dist = g.distances(&truth, goal);
        let edist = eg.distances(&etruth, goal);
        let target = g.targets(&truth, &dist, goal);
        let etarget = eg.targets(&etruth, &edist, goal);
        Self {
            m,
            g,
            eg,
            goal,
            truth,
            etruth,
            dist,
            edist,
            target,
            etarget,
            changed,
        }
    }
    fn graph(&self, k: Kind) -> &Graph {
        if k == Kind::EdgeDiagonal {
            &self.eg
        } else {
            &self.g
        }
    }
}
struct Tape {
    v: Vec<f64>,
    p: Vec<f64>,
    n: usize,
    h: usize,
}
fn forward(g: &Graph, c: &[f64], goal: usize) -> Tape {
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
fn loss(case: &Case, w: &[f64], kind: Kind, gradient: bool) -> (f64, Vec<f64>) {
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
struct Adam {
    m: [f64; P],
    v: [f64; P],
    t: i32,
}
impl Adam {
    fn new() -> Self {
        Self {
            m: [0.0; P],
            v: [0.0; P],
            t: 0,
        }
    }
    fn step(&mut self, w: &mut [f64], g: &[f64], kind: Kind) {
        self.t += 1;
        let norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        let clip = if norm > 10.0 { 10.0 / norm } else { 1.0 };
        for j in 0..P {
            if !kind.trainable(j) {
                continue;
            }
            let d = g[j] * clip;
            self.m[j] = 0.9 * self.m[j] + 0.1 * d;
            self.v[j] = 0.999 * self.v[j] + 0.001 * d * d;
            w[j] -= 0.03 * (self.m[j] / (1.0 - 0.9f64.powi(self.t)))
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
#[derive(Default)]
struct Metrics {
    nll: f64,
    accuracy: f64,
    success: f64,
    regret: f64,
    interior_fraction: f64,
    maps: usize,
    starts: usize,
    states: usize,
    edges: usize,
    eval_ms: f64,
}
fn route(
    case: &Case,
    c: &[f64],
    t: Option<&Tape>,
    kind: Kind,
    start: usize,
) -> (Vec<usize>, f64, bool, usize, usize) {
    let g = case.graph(kind);
    let truth = if kind == Kind::EdgeDiagonal {
        &case.etruth
    } else {
        &case.truth
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
fn evaluate(cases: &[Case], w: &[f64], kind: Kind) -> Metrics {
    let start = Instant::now();
    let mut z = Metrics::default();
    for case in cases {
        let g = case.graph(kind);
        let c = if kind == Kind::Oracle {
            case.truth.clone()
        } else {
            costs(&case.m, g, w, kind, false).0
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
                let tar = if kind == Kind::EdgeDiagonal {
                    case.etarget[x]
                } else {
                    case.target[x]
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
            let (_, cost, ok, interior, steps) = route(case, &c, t.as_ref(), kind, x);
            let denom = (g.n_original - 1) as f64;
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
fn diagnostics(cases: &[Case], w: &[f64]) -> String {
    let mut out=String::from("map,vertices,edges,faces,tetrahedra,steiner_states,steiner_actions,interior_hinges,regge_energy,max_deficit,min_volume,min_cm_volume2,min_transition,refined_states,refined_actions,coarse_oracle_mean,refined_oracle_mean,edge_oracle_mean,hodge_symmetry_error,hodge_energy\n");
    for (i, c) in cases.iter().take(3).enumerate() {
        assert!(c.m.boundary_checks());
        let (h, e, d) = c.m.regge(w);
        let v =
            c.m.tets
                .iter()
                .map(|t| volume(t.map(|j| c.m.p[j]), w))
                .fold(f64::INFINITY, f64::min);
        let cm =
            c.m.tets
                .iter()
                .map(|t| cm_volume2(t.map(|j| c.m.p[j]), w))
                .fold(f64::INFINITY, f64::min);
        let ct = costs(&c.m, &c.g, w, Kind::Full, false)
            .0
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let rg = Graph::new(&c.m, false, true);
        let rc = costs(&c.m, &rg, &TEACHER, Kind::Full, c.changed).0;
        let rd = rg.distances(&rc, c.goal);
        let mean = |ds: &[f64]| {
            (0..c.m.p.len())
                .filter(|&j| j != c.goal)
                .map(|j| ds[j])
                .sum::<f64>()
                / (c.m.p.len() - 1) as f64
        };
        let (m0, m1, m2) = c.m.hodge_weights(w);
        let a: Vec<_> = (0..m1.len()).map(|i| (i as f64 * 0.41).sin()).collect();
        let b: Vec<_> = (0..m1.len()).map(|i| (i as f64 * 0.17).cos()).collect();
        let la = c.m.hodge1(&a, &m0, &m1, &m2);
        let lb = c.m.hodge1(&b, &m0, &m1, &m2);
        let lhs: f64 = (0..a.len()).map(|i| a[i] * m1[i] * lb[i]).sum();
        let rhs: f64 = (0..a.len()).map(|i| la[i] * m1[i] * b[i]).sum();
        let en: f64 = (0..a.len()).map(|i| a[i] * m1[i] * la[i]).sum();
        assert!(mean(&rd) <= mean(&c.dist) + 1e-9);
        assert!(mean(&c.dist) <= mean(&c.edist) + 1e-9);
        out+=&format!("{i},{},{},{},{},{},{},{h},{e:0.12},{d:0.12},{v:0.12},{cm:0.12},{ct:0.12},{},{},{:0.12},{:0.12},{:0.12},{:0.12},{en:0.12}\n",c.m.p.len(),c.m.edges.len(),c.m.faces.len(),c.m.tets.len(),c.g.points.len(),c.g.edges.len(),rg.points.len(),rg.edges.len(),mean(&c.dist),mean(&rd),mean(&c.edist),(lhs-rhs).abs());
    }
    out
}
fn train(seed: u64, phase: &str, dir: &str) {
    let clock = Instant::now();
    fs::create_dir_all(dir).unwrap();
    let tr = dataset(seed, 0x11100000000, 12, [2, 1, 1], 0.06, false);
    let val = dataset(seed, 0x22200000000, 4, [2, 1, 1], 0.06, false);
    let test = dataset(seed, 0x33300000000, 24, [2, 1, 1], 0.06, false);
    let ood = dataset(seed, 0x44400000000, 8, [2, 2, 1], 0.09, false);
    let changed = dataset(seed, 0x55500000000, 8, [2, 1, 1], 0.06, true);
    let mut r = Rng::new(seed ^ 0x66600000000);
    let init: Vec<f64> = (0..P).map(|_| 0.1 * r.normal()).collect();
    let mut weights = vec![init.clone(); 4];
    for i in [1, 2] {
        weights[i][3..6].fill(0.0);
    }
    let mut opts: Vec<_> = (0..4).map(|_| Adam::new()).collect();
    let mut log = BufWriter::new(File::create(format!("{dir}/epochs.csv")).unwrap());
    writeln!(
        log,
        "epoch,full_nll,diagonal_nll,edge_diagonal_nll,no_phi_nll,seconds"
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
        writeln!(
            log,
            "{ep},{:0.10},{:0.10},{:0.10},{:0.10},{:0.4}",
            ls[0],
            ls[1],
            ls[2],
            ls[3],
            clock.elapsed().as_secs_f64()
        )
        .unwrap();
        if ep % 20 == 0 {
            log.flush().unwrap();
            println!(
                "{phase} {seed:X} epoch={ep} losses={ls:?} seconds={:0.2}",
                clock.elapsed().as_secs_f64()
            );
        }
        assert!(clock.elapsed().as_secs() < 2700);
    }
    log.flush().unwrap();
    let mut out = BufWriter::new(File::create(format!("{dir}/metrics.csv")).unwrap());
    writeln!(out,"phase,seed,split,model,nll,accuracy,success,regret,interior_fraction,maps,starts,states,actions,eval_ms").unwrap();
    for (split, cases) in [
        ("train", &tr),
        ("val", &val),
        ("test", &test),
        ("ood_geometry", &ood),
        ("changed_law", &changed),
    ] {
        for i in 0..7 {
            let kind = if i < 4 {
                TRAIN_KINDS[i]
            } else {
                [Kind::Untrained, Kind::Euclidean, Kind::Oracle][i - 4]
            };
            let w = if i < 4 { &weights[i] } else { &init };
            let m = evaluate(cases, w, kind);
            writeln!(out,"{phase},{seed},{split},{},{:0.10},{:0.10},{:0.10},{:0.10},{:0.10},{},{},{},{},{:0.4}",kind.name(),m.nll,m.accuracy,m.success,m.regret,m.interior_fraction,m.maps,m.starts,m.states,m.edges,m.eval_ms).unwrap();
        }
    }
    out.flush().unwrap();
    let mut text = format!("initial={init:?}\n");
    for i in 0..4 {
        text += &format!("{}={:?}\n", TRAIN_KINDS[i].name(), weights[i]);
    }
    text += &format!("seconds={}\n", clock.elapsed().as_secs_f64());
    fs::write(format!("{dir}/weights.txt"), text).unwrap();
    fs::write(
        format!("{dir}/geometry.csv"),
        diagnostics(&test, &weights[0]),
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
            text+=&format!("{} start={start} path={path:?} coords={coords:?} true_cost={real:0.10} oracle={:0.10} success={ok} interior_segments={inside}/{steps}\n",k.name(),c.dist[start]);
        }
    }
    fs::write(format!("{dir}/routes.txt"), text).unwrap();
    println!(
        "DONE {phase} {seed:X} seconds={:0.3}",
        clock.elapsed().as_secs_f64()
    );
}
fn main() {
    let a: Vec<_> = env::args().collect();
    if a.len() != 5 || a[1] != "train" {
        eprintln!("usage: soft-bellman-simplicial-v2 train DECIMAL_SEED PHASE OUTPUT_DIR");
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
        // A randomly blocked mesh may have no interior hinges. Use the complete
        // cube to verify the non-vacuous interior deficit and boundary exclusion.
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
}
