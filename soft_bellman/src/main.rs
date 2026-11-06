//! Restricted, finite-horizon Soft-Bellman inverse-planning experiment.
//! All training and analytic reverse differentiation run in Rust, without dependencies.
use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    time::Instant,
};
const TAU: f64 = 0.2;
const EPS: f64 = 0.01;
const P: usize = 9;
const NF: usize = 8;
const NH: usize = 8;
const MP: usize = NF * NH + NH + NH + 1;
const EPOCHS: usize = 160;

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
#[derive(Clone, Debug)]
struct Edge {
    src: usize,
    dst: usize,
    axis: usize,
}
#[derive(Clone)]
struct Map {
    side: usize,
    dim: usize,
    n: usize,
    goal: usize,
    wall: Vec<bool>,
    f: Vec<[f64; 3]>,
    edges: Vec<Edge>,
    out: Vec<Vec<usize>>,
    oracle: Vec<f64>,
    target: Vec<usize>,
    truth: Vec<f64>,
}
impl Map {
    fn coord(&self, x: usize, d: usize) -> usize {
        (x / self.side.pow(d as u32)) % self.side
    }
    fn manhattan(&self, x: usize) -> usize {
        (0..self.dim)
            .map(|d| self.coord(x, d).abs_diff(self.coord(self.goal, d)))
            .sum()
    }
    fn generate(rng: &mut Rng, side: usize, dim: usize) -> Self {
        loop {
            let n = side.pow(dim as u32);
            let goal = rng.usize(n);
            let mut wall: Vec<bool> = (0..n).map(|_| rng.uniform() < 0.18).collect();
            wall[goal] = false;
            let f = (0..n)
                .map(|_| [1.0, rng.uniform(), rng.uniform()])
                .collect();
            let mut m = Self {
                side,
                dim,
                n,
                goal,
                wall,
                f,
                edges: vec![],
                out: vec![vec![]; n],
                oracle: vec![f64::INFINITY; n],
                target: vec![usize::MAX; n],
                truth: vec![],
            };
            for x in 0..n {
                if m.wall[x] {
                    continue;
                }
                for d in 0..dim {
                    let stride = side.pow(d as u32);
                    let c = m.coord(x, d);
                    for sign in [-1isize, 1] {
                        if (sign < 0 && c == 0) || (sign > 0 && c + 1 == side) {
                            continue;
                        }
                        let y = (x as isize + sign * stride as isize) as usize;
                        if !m.wall[y] {
                            let id = m.edges.len();
                            m.out[x].push(id);
                            m.edges.push(Edge {
                                src: x,
                                dst: y,
                                axis: usize::from(d > 0),
                            });
                        }
                    }
                }
            }
            // Teacher parameters stay exclusively in dataset labeling/evaluation.
            let teacher = [0.3, 1.6, -0.8, 0.1, -0.6, 1.8, -0.8, 1.4, 1.0];
            m.truth = m.costs(&teacher).0;
            m.oracle = m.distances(&m.truth);
            if (0..n).any(|x| !m.wall[x] && !m.oracle[x].is_finite())
                || n - m.wall.iter().filter(|x| **x).count() < n / 2
            {
                continue;
            }
            for x in 0..n {
                if x == goal || m.wall[x] {
                    continue;
                }
                m.target[x] = *m.out[x]
                    .iter()
                    .min_by(|&&a, &&b| {
                        (m.truth[a] + m.oracle[m.edges[a].dst])
                            .total_cmp(&(m.truth[b] + m.oracle[m.edges[b].dst]))
                    })
                    .unwrap();
            }
            return m;
        }
    }
    fn distances(&self, cost: &[f64]) -> Vec<f64> {
        let mut d = vec![f64::INFINITY; self.n];
        d[self.goal] = 0.0;
        let mut used = vec![false; self.n];
        for _ in 0..self.n {
            let x = (0..self.n)
                .filter(|&x| !used[x])
                .min_by(|&a, &b| d[a].total_cmp(&d[b]))
                .unwrap();
            if !d[x].is_finite() {
                break;
            }
            used[x] = true;
            for (id, e) in self.edges.iter().enumerate() {
                if e.dst == x {
                    d[e.src] = d[e.src].min(d[x] + cost[id]);
                }
            }
        }
        d
    }
    fn costs(&self, w: &[f64]) -> (Vec<f64>, Vec<[f64; P]>) {
        let mut c = Vec::with_capacity(self.edges.len());
        let mut dc = Vec::with_capacity(self.edges.len());
        for e in &self.edges {
            let k = e.axis * 3;
            let a = dot(&w[k..k + 3], &self.f[e.src]);
            let b = dot(&w[6..9], &self.f[e.dst]);
            let s = sp(a);
            let metric = (EPS + s * s).sqrt();
            c.push(1.0 + metric + sp(b));
            let mut g = [0.0; P];
            for j in 0..3 {
                g[k + j] = s / metric * sig(a) * self.f[e.src][j];
                g[6 + j] = sig(b) * self.f[e.dst][j];
            }
            dc.push(g);
        }
        (c, dc)
    }
    fn features(&self, id: usize) -> [f64; NF] {
        let e = &self.edges[id];
        let denom = self.side as f64;
        [
            self.f[e.src][1],
            self.f[e.src][2],
            self.f[e.dst][1],
            self.f[e.dst][2],
            (self.manhattan(e.src) as f64 - self.manhattan(e.dst) as f64),
            e.axis as f64,
            self.manhattan(e.dst) as f64 / (denom * self.dim as f64),
            f64::from(e.dst == self.goal),
        ]
    }
    fn states(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n).filter(|&x| x != self.goal && !self.wall[x] && self.oracle[x].is_finite())
    }
}
struct Tape {
    v: Vec<f64>,
    p: Vec<f64>,
    h: usize,
}
fn forward(m: &Map, c: &[f64], h: usize) -> Tape {
    let mut v = vec![f64::INFINITY; (h + 1) * m.n];
    let mut p = vec![0.0; (h + 1) * m.edges.len()];
    v[m.goal] = 0.0;
    for t in 1..=h {
        v[t * m.n + m.goal] = 0.0;
        for x in 0..m.n {
            if x == m.goal || m.wall[x] {
                continue;
            }
            let q: Vec<f64> = m.out[x]
                .iter()
                .map(|&id| c[id] + v[(t - 1) * m.n + m.edges[id].dst])
                .collect();
            let (val, pr) = softmin(&q);
            v[t * m.n + x] = val;
            for (&id, prob) in m.out[x].iter().zip(pr) {
                p[t * m.edges.len() + id] = prob;
            }
        }
    }
    Tape { v, p, h }
}
fn field_loss(m: &Map, w: &[f64], gradient: bool) -> (f64, Vec<f64>) {
    let (c, dc) = m.costs(w);
    let tape = forward(m, &c, m.n);
    let ne = m.edges.len();
    let count = m.states().count() as f64;
    let mut loss = 0.0;
    let mut gc = vec![0.0; ne];
    let mut av = vec![0.0; tape.v.len()];
    let t = tape.h;
    for x in m.states() {
        let target = m.target[x];
        let y = m.edges[target].dst;
        loss += (c[target] + tape.v[(t - 1) * m.n + y] - tape.v[t * m.n + x]) / TAU / count;
        if gradient {
            for &id in &m.out[x] {
                let e = &m.edges[id];
                let q = (f64::from(id == target) - tape.p[t * ne + id]) / TAU / count;
                gc[id] += q;
                av[(t - 1) * m.n + e.dst] += q;
            }
        }
    }
    if gradient {
        for layer in (1..t).rev() {
            for x in m.states() {
                let adj = av[layer * m.n + x];
                if adj == 0.0 {
                    continue;
                }
                for &id in &m.out[x] {
                    let e = &m.edges[id];
                    let d = adj * tape.p[layer * ne + id];
                    gc[id] += d;
                    av[(layer - 1) * m.n + e.dst] += d;
                }
            }
        }
    }
    let mut g = vec![0.0; P];
    if gradient {
        for id in 0..ne {
            for j in 0..P {
                g[j] += gc[id] * dc[id][j];
            }
        }
    }
    (loss, g)
}
fn mlp_score(w: &[f64], f: &[f64; NF]) -> (f64, [f64; NH]) {
    let mut h = [0.0; NH];
    for k in 0..NH {
        h[k] = (dot(&w[k * NF..(k + 1) * NF], f) + w[NF * NH + k]).tanh();
    }
    let score = dot(&w[NF * NH + NH..NF * NH + 2 * NH], &h) + w[MP - 1];
    (score, h)
}
fn mlp_loss(m: &Map, w: &[f64], gradient: bool) -> (f64, Vec<f64>) {
    let mut loss = 0.0;
    let mut g = vec![0.0; MP];
    let n = m.states().count() as f64;
    for x in m.states() {
        let items: Vec<_> = m.out[x]
            .iter()
            .map(|&id| {
                let f = m.features(id);
                let (s, h) = mlp_score(w, &f);
                (id, f, s, h)
            })
            .collect();
        let max = items.iter().map(|a| a.2).fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = items.iter().map(|a| (a.2 - max).exp()).sum();
        for (id, f, s, h) in items {
            if id == m.target[x] {
                loss += (max + z.ln() - s) / n;
            }
            if gradient {
                let d = ((s - max).exp() / z - f64::from(id == m.target[x])) / n;
                g[MP - 1] += d;
                for k in 0..NH {
                    g[NF * NH + NH + k] += d * h[k];
                    let a = d * w[NF * NH + NH + k] * (1.0 - h[k] * h[k]);
                    g[NF * NH + k] += a;
                    for j in 0..NF {
                        g[k * NF + j] += a * f[j];
                    }
                }
            }
        }
    }
    (loss, g)
}
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}
impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }
    fn step(&mut self, w: &mut [f64], g: &[f64]) {
        self.t += 1;
        let norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = if norm > 10.0 { 10.0 / norm } else { 1.0 };
        for i in 0..w.len() {
            let d = g[i] * scale;
            assert!(d.is_finite());
            self.m[i] = 0.9 * self.m[i] + 0.1 * d;
            self.v[i] = 0.999 * self.v[i] + 0.001 * d * d;
            w[i] -= 0.03 * (self.m[i] / (1.0 - 0.9f64.powi(self.t)))
                / ((self.v[i] / (1.0 - 0.999f64.powi(self.t))).sqrt() + 1e-8);
            assert!(w[i].is_finite());
        }
    }
}
fn batch(maps: &[Map], w: &[f64], mlp: bool, grad: bool) -> (f64, Vec<f64>) {
    let mut l = 0.0;
    let mut g = vec![0.0; w.len()];
    for m in maps {
        let (a, b) = if mlp {
            mlp_loss(m, w, grad)
        } else {
            field_loss(m, w, grad)
        };
        l += a / maps.len() as f64;
        for (j, v) in g.iter_mut().enumerate() {
            *v += b[j] / maps.len() as f64;
        }
    }
    assert!(l.is_finite());
    (l, g)
}
#[derive(Default, Clone)]
struct Metrics {
    nll: f64,
    acc: f64,
    success: f64,
    regret: f64,
    maps: usize,
    states: usize,
    eval_ms: f64,
}
fn evaluate(maps: &[Map], w: &[f64], kind: &str) -> Metrics {
    let start = Instant::now();
    let mut z = Metrics::default();
    for m in maps {
        let c = if kind == "constant" {
            vec![1.0; m.edges.len()]
        } else {
            m.costs(w).0
        };
        let tape = if kind != "mlp" {
            Some(forward(m, &c, m.n))
        } else {
            None
        };
        let mut a = Metrics::default();
        let n = m.states().count() as f64;
        for x in m.states() {
            let eid = if let Some(ref t) = tape {
                let target = m.target[x];
                a.nll += (c[target] + t.v[(m.n - 1) * m.n + m.edges[target].dst]
                    - t.v[m.n * m.n + x])
                    / TAU
                    / n;
                *m.out[x]
                    .iter()
                    .min_by(|&&i, &&j| {
                        (c[i] + t.v[(m.n - 1) * m.n + m.edges[i].dst])
                            .total_cmp(&(c[j] + t.v[(m.n - 1) * m.n + m.edges[j].dst]))
                    })
                    .unwrap()
            } else {
                let items: Vec<_> = m.out[x]
                    .iter()
                    .map(|&id| (id, mlp_score(w, &m.features(id)).0))
                    .collect();
                let max = items.iter().map(|a| a.1).fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = items.iter().map(|a| (a.1 - max).exp()).sum();
                let target = items.iter().find(|a| a.0 == m.target[x]).unwrap().1;
                a.nll += (max + sum.ln() - target) / n;
                items.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0
            };
            if eid == m.target[x] {
                a.acc += 1.0 / n;
            }
            let (route, cost, ok) = route(m, &c, tape.as_ref(), w, x, kind);
            let _ = route;
            a.success += f64::from(ok) / n;
            a.regret += if ok {
                (cost / m.oracle[x] - 1.0).max(0.0) / n
            } else {
                5.0 / n
            };
        }
        z.nll += a.nll / maps.len() as f64;
        z.acc += a.acc / maps.len() as f64;
        z.success += a.success / maps.len() as f64;
        z.regret += a.regret / maps.len() as f64;
        z.states += n as usize;
        z.maps += 1;
    }
    z.eval_ms = start.elapsed().as_secs_f64() * 1000.0;
    z
}
fn route(
    m: &Map,
    c: &[f64],
    tape: Option<&Tape>,
    w: &[f64],
    start: usize,
    kind: &str,
) -> (Vec<usize>, f64, bool) {
    let mut x = start;
    let mut cost = 0.0;
    let mut path = vec![x];
    for rem in (1..=m.n).rev() {
        if x == m.goal {
            return (path, cost, true);
        }
        let id = if kind == "mlp" {
            *m.out[x]
                .iter()
                .max_by(|&&i, &&j| {
                    mlp_score(w, &m.features(i))
                        .0
                        .total_cmp(&mlp_score(w, &m.features(j)).0)
                })
                .unwrap()
        } else {
            let t = tape.unwrap();
            let best = m.out[x].iter().min_by(|&&i, &&j| {
                (c[i] + t.v[(rem - 1) * m.n + m.edges[i].dst])
                    .total_cmp(&(c[j] + t.v[(rem - 1) * m.n + m.edges[j].dst]))
            });
            match best {
                Some(&id) => id,
                None => return (path, cost, false),
            }
        };
        cost += m.truth[id];
        x = m.edges[id].dst;
        path.push(x);
    }
    let ok = x == m.goal;
    (path, cost, ok)
}
fn dataset(seed: u64, tag: u64, count: usize, side: usize, dim: usize) -> Vec<Map> {
    let mut r = Rng::new(seed ^ tag);
    (0..count)
        .map(|_| Map::generate(&mut r, side, dim))
        .collect()
}
fn write_row(out: &mut impl Write, seed: u64, phase: &str, split: &str, model: &str, m: &Metrics) {
    writeln!(
        out,
        "{phase},{seed},{split},{model},{:.10},{:.10},{:.10},{:.10},{},{},{:.4}",
        m.nll, m.acc, m.success, m.regret, m.maps, m.states, m.eval_ms
    )
    .unwrap();
}
fn train(seed: u64, phase: &str, dir: &str) {
    let start = Instant::now();
    fs::create_dir_all(dir).unwrap();
    let train = dataset(seed, 0x10100000000, 24, 6, 2);
    let val = dataset(seed, 0x20200000000, 8, 6, 2);
    let test = dataset(seed, 0x30300000000, 64, 6, 2);
    let ood = dataset(seed, 0x40400000000, 16, 3, 3);
    let mut rng = Rng::new(seed ^ 0x50500000000);
    let mut w: Vec<f64> = (0..P).map(|_| 0.15 * rng.normal()).collect();
    let init = w.clone();
    let mut mw: Vec<f64> = (0..MP).map(|_| 0.15 * rng.normal()).collect();
    let mut opt = Adam::new(P);
    let mut mo = Adam::new(MP);
    let mut log = BufWriter::new(File::create(format!("{dir}/epochs.csv")).unwrap());
    writeln!(log, "epoch,field_train_nll,mlp_train_nll,seconds").unwrap();
    for epoch in 0..=EPOCHS {
        let (l, g) = batch(&train, &w, false, true);
        let (ml, mg) = batch(&train, &mw, true, true);
        writeln!(
            log,
            "{epoch},{l:.10},{ml:.10},{:.4}",
            start.elapsed().as_secs_f64()
        )
        .unwrap();
        if epoch % 20 == 0 {
            log.flush().unwrap();
            println!(
                "{phase} {seed:x} epoch={epoch} nll={l:.5} mlp={ml:.5} sec={:.2}",
                start.elapsed().as_secs_f64()
            );
        }
        if epoch == EPOCHS {
            break;
        }
        opt.step(&mut w, &g);
        mo.step(&mut mw, &mg);
        assert!(start.elapsed().as_secs() < 2700, "45-minute stop rule");
    }
    log.flush().unwrap();
    let mut out = BufWriter::new(File::create(format!("{dir}/metrics.csv")).unwrap());
    writeln!(
        out,
        "phase,seed,split,model,nll,accuracy,success,regret,maps,states,eval_ms"
    )
    .unwrap();
    for (split, maps) in [
        ("train", &train),
        ("val", &val),
        ("test", &test),
        ("ood3d", &ood),
    ] {
        for (model, weights) in [
            ("field", &w),
            ("untrained", &init),
            ("constant", &init),
            ("mlp", &mw),
        ] {
            let m = evaluate(maps, weights, model);
            write_row(&mut out, seed, phase, split, model, &m);
        }
    }
    out.flush().unwrap();
    fs::write(format!("{dir}/weights.txt"),format!("field_initial={init:?}\nfield_final={w:?}\nmlp_final={mw:?}\nparameters_field={P}\nparameters_mlp={MP}\nseconds={}\n",start.elapsed().as_secs_f64())).unwrap();
    let mut routes = String::new();
    for (k, m) in test.iter().take(3).enumerate() {
        routes.push_str(&format!(
            "map={k} side={} dim={} goal={} blocked={:?}\nterrain={:?}\n",
            m.side,
            m.dim,
            m.goal,
            m.wall
                .iter()
                .enumerate()
                .filter(|(_, b)| **b)
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            m.f
        ));
        let x = m.states().max_by_key(|&x| m.manhattan(x)).unwrap();
        for (model, weights) in [
            ("field", &w),
            ("untrained", &init),
            ("constant", &init),
            ("mlp", &mw),
        ] {
            let c = if model == "constant" {
                vec![1.0; m.edges.len()]
            } else {
                m.costs(weights).0
            };
            let tape = forward(m, &c, m.n);
            let (p, cost, ok) = route(m, &c, Some(&tape), weights, x, model);
            routes.push_str(&format!("{model}: start={x} path={p:?} true_cost={cost:.8} oracle_cost={:.8} success={ok}\n",m.oracle[x]));
        }
    }
    fs::write(format!("{dir}/routes.txt"), routes).unwrap();
    println!(
        "DONE {phase} {seed:x} seconds={:.3}",
        start.elapsed().as_secs_f64()
    );
}
fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 || args[1] != "train" {
        eprintln!("usage: soft-bellman-v1 train SEED_INTEGER DEV|CONFIRM OUTPUT_DIR");
        std::process::exit(2);
    }
    let seed = args[2].parse().unwrap();
    train(seed, &args[3], &args[4]);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn softmin_extremes() {
        let (v, p) = softmin(&[1e6, 1e6 + 1.0, f64::INFINITY]);
        assert!(v.is_finite());
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(p[2], 0.0);
        let (v, p) = softmin(&[f64::INFINITY]);
        assert!(v.is_infinite());
        assert_eq!(p[0], 0.0);
    }
    #[test]
    fn positive_spd() {
        let mut r = Rng::new(3);
        let m = Map::generate(&mut r, 4, 3);
        let w = [-100.0; P];
        let (c, _) = m.costs(&w);
        assert!(c.iter().all(|x| *x >= 1.1 - 1e-12));
        for e in &m.edges {
            let s = sp(dot(&w[e.axis * 3..e.axis * 3 + 3], &m.f[e.src]));
            assert!(EPS + s * s > 0.0);
        }
    }
    #[test]
    fn absorbing_goal_and_oracle() {
        let mut r = Rng::new(7);
        let m = Map::generate(&mut r, 4, 2);
        let t = forward(&m, &m.truth, m.n);
        for h in 0..=m.n {
            assert_eq!(t.v[h * m.n + m.goal], 0.0);
        }
        for x in m.states() {
            let e = m.target[x];
            assert!((m.truth[e] + m.oracle[m.edges[e].dst] - m.oracle[x]).abs() < 1e-10);
            assert!(t.v[m.n * m.n + x].is_finite());
        }
    }
    #[test]
    fn unreachable() {
        let mut r = Rng::new(7);
        let mut m = Map::generate(&mut r, 4, 2);
        let x = m.states().next().unwrap();
        m.out[x].clear();
        let t = forward(&m, &m.truth, m.n);
        assert!(t.v[m.n * m.n + x].is_infinite());
    }
    #[test]
    fn generic_3d() {
        let maps = dataset(4, 8, 2, 3, 3);
        let w = [0.0; P];
        let m = evaluate(&maps, &w, "field");
        assert_eq!(m.maps, 2);
        assert!((m.success - 1.0).abs() < 1e-12);
        assert!(m.regret.is_finite());
    }
    #[test]
    fn deterministic_replay() {
        let a = dataset(8, 9, 2, 4, 2);
        let b = dataset(8, 9, 2, 4, 2);
        assert_eq!(a[0].f, b[0].f);
        assert_eq!(a[1].wall, b[1].wall);
        assert_eq!(
            field_loss(&a[0], &[0.1; P], true),
            field_loss(&b[0], &[0.1; P], true)
        );
    }
    #[test]
    fn field_gradients() {
        let m = &dataset(52, 9, 1, 4, 2)[0];
        let mut w = vec![0.1; P];
        let (_, g) = field_loss(m, &w, true);
        for j in 0..P {
            let old = w[j];
            w[j] = old + 1e-5;
            let a = field_loss(m, &w, false).0;
            w[j] = old - 1e-5;
            let b = field_loss(m, &w, false).0;
            w[j] = old;
            let fd = (a - b) / 2e-5;
            assert!(
                (fd - g[j]).abs() < 1e-5 * (1.0 + fd.abs()),
                "j={j}, fd={fd}, grad={}",
                g[j]
            );
        }
    }
    #[test]
    fn mlp_gradients() {
        let m = &dataset(52, 9, 1, 4, 2)[0];
        let mut r = Rng::new(3);
        let mut w: Vec<f64> = (0..MP).map(|_| 0.15 * r.normal()).collect();
        let (_, g) = mlp_loss(m, &w, true);
        for j in 0..MP {
            let old = w[j];
            w[j] = old + 1e-5;
            let a = mlp_loss(m, &w, false).0;
            w[j] = old - 1e-5;
            let b = mlp_loss(m, &w, false).0;
            w[j] = old;
            let fd = (a - b) / 2e-5;
            assert!(
                (fd - g[j]).abs() < 1e-5 * (1.0 + fd.abs()),
                "j={j}, fd={fd}, grad={}",
                g[j]
            );
        }
    }
    #[test]
    fn learning_smoke() {
        let maps = dataset(52, 9, 2, 4, 2);
        let mut w = vec![0.1; P];
        let mut o = Adam::new(P);
        let initial = batch(&maps, &w, false, false).0;
        for _ in 0..30 {
            let (_, g) = batch(&maps, &w, false, true);
            o.step(&mut w, &g);
        }
        assert!(batch(&maps, &w, false, false).0 < initial);
    }
}
