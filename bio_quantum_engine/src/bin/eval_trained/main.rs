//! Evaluación SOLO-INFERENCIA de los pesos entrenados en GPU (weights_x50.bin / weights_x20.bin)
//! con el protocolo de `eval` de la rama (mismos generadores, semillas, métricas).
//! Port CPU f32 del kernel `forward_pass` (WGSL). Pesos congelados: no se llama a train_step ni SWR.
//! Formato .bin (gpu-train): f32 LE, [wp_exc(2d) | wp_inh(2d)]; wf_exc = wf_inh = 1e-4 (init).
mod w {
    pub fn sqrt(a: f32) -> f32 { a.sqrt() }
    pub fn max(a: f32, b: f32) -> f32 { a.max(b) }
    pub fn atan2(y: f32, x: f32) -> f32 { y.atan2(x) }
    pub fn tanh(a: f32) -> f32 { a.tanh() }
    pub fn exp(a: f32) -> f32 { a.exp() }
    pub fn sin(a: f32) -> f32 { a.sin() }
    pub fn cos(a: f32) -> f32 { a.cos() }
    pub fn log(a: f32) -> f32 { a.ln() }
    pub fn abs(a: f32) -> f32 { a.abs() }
    pub fn atan(a: f32) -> f32 { a.atan() }
    pub fn sinh(a: f32) -> f32 { a.sinh() }
    pub fn clamp(a: f32, l: f32, h: f32) -> f32 { a.max(l).min(h) }
    pub fn f32c(a: u32) -> f32 { a as f32 }
}
mod phi_gen;
use bio_quantum_engine::protocol_cst4::{compute_log_relative_error, compute_relative_error, Cst4DataGenerator, HORIZON, K_OBS, NEW_FAMILIES, REL_ERR_OK};
use bio_quantum_engine::rng::Xoshiro256StarStar;
use std::io::Read;

struct Model { d: usize, wx: Vec<f32>, wy: Vec<f32>, r_bound: f32, phi: fn(f32, f32, u32) -> f32, raw: Vec<f32>, persist: bool }

impl Model {
    fn from_raw(raw: Vec<f32>, persist: bool) -> Self {
        let d = raw.len() / 4;
        let phi: fn(f32, f32, u32) -> f32 = if d == 1024 { phi_gen::compute_phi_1024 } else if d == 512 { phi_gen::compute_phi_512 } else { panic!("d={} no soportado", d) };
        let (we, wi) = raw.split_at(2 * d);
        let wf = 1e-4f32 - 1e-4f32;
        let wx = (0..d).map(|j| (we[j] - wi[j]) + wf).collect();
        let wy = (0..d).map(|j| (we[d + j] - wi[d + j]) + wf).collect();
        Model { d, wx, wy, r_bound: 15.0, phi, raw, persist }
    }
    fn step(&self, f: &[f64]) -> Vec<f64> {
        if self.persist { return f.to_vec(); }
        let mut o = vec![0.0; f.len()];
        for p in 0..f.len() / 2 {
            let (x, y) = (f[2 * p] as f32, f[2 * p + 1] as f32);
            let (mut dx, mut dy) = (0f32, 0f32);
            for j in 0..self.d { let ph = (self.phi)(x, y, j as u32); dx += self.wx[j] * ph; dy += self.wy[j] * ph; }
            let (mut px, mut py) = (x + dx, y + dy);
            let rb = self.r_bound; let rp = (px * px + py * py).max(1e-12).sqrt();
            if rp > rb { let s = rb * (1.0 + ((rp - rb) / rb).tanh()) / rp; px *= s; py *= s; }
            o[2 * p] = px as f64; o[2 * p + 1] = py as f64;
        }
        o
    }
    fn rollout(&self, q: &[f64], h: usize) -> Vec<Vec<f64>> { let mut c = q.to_vec(); (0..h).map(|_| { c = self.step(&c); c.clone() }).collect() }
}

fn eval_seed(m: &Model, seed: u64, n: usize) -> (f64, f64, f64, f64, bool, Vec<f64>) {
    let g = Cst4DataGenerator::new(); let t = K_OBS + HORIZON;
    let (mut ka, mut na) = (0, 0); let (mut kl, mut nl) = (vec![], vec![]);
    for ep in 0..n {
        let mut r = Xoshiro256StarStar::from_seed(seed ^ (0xFEEDFACE + ep as u64));
        for (fam, acc, logs) in [(ep % 8, &mut ka, &mut kl), (NEW_FAMILIES[ep % NEW_FAMILIES.len()], &mut na, &mut nl)] {
            let e = g.generate_episode(fam, t, &mut r);
            let p = m.rollout(&e.trajectory[K_OBS - 1], HORIZON);
            let (pf, tf) = (&p[HORIZON - 1], &e.trajectory[t - 1]);
            if compute_relative_error(pf, tf) < REL_ERR_OK { *acc += 1; }
            logs.push(compute_log_relative_error(pf, tf));
        }
    }
    let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let mut e33 = vec![];
    for &h in &[1usize, 2, 4, 8, 16] {
        let mut s = 0; let mut r = Xoshiro256StarStar::from_seed(seed ^ 0x33);
        for ep in 0..n { let e = g.generate_episode(ep % 8, K_OBS + h, &mut r);
            let p = m.rollout(&e.trajectory[K_OBS - 1], h);
            if compute_relative_error(&p[h - 1], &e.trajectory[K_OBS + h - 1]) < REL_ERR_OK { s += 1; } }
        e33.push(s as f64 / n as f64);
    }
    let pass = e33.iter().all(|&a| a >= 0.80);
    (ka as f64 / n as f64, med(&mut kl), na as f64 / n as f64, med(&mut nl), pass, e33)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (wpath, phase) = (&a[1], a[2].as_str());
    let persist = wpath == "persistence";
    let raw: Vec<f32> = if persist { vec![0.0; 4096] } else {
        let mut b = vec![]; std::fs::File::open(wpath).unwrap().read_to_end(&mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let m = Model::from_raw(raw, persist);
    let viol = m.raw.iter().filter(|&&w| w < 0.0).count();
    let below = m.raw.iter().filter(|&&w| w < 1e-4 - 1e-9).count();
    if phase == "trainloss" {
        // Chequeo de fidelidad del port: MSE de 1 paso sobre la distribución de gpu-train (lote 400, 64 items)
        let np = m.d; let (mut l, mut c) = (0f64, 0usize);
        for b in 0..64 { let ph = 400f32 * 0.05 + b as f32 * 0.1; let rad = 1.0 + 0.3 * (ph * 0.5).sin();
            let mut f = vec![]; let mut t = vec![];
            for q in 0..np { let th = q as f32 * std::f32::consts::PI * 2.0 / np as f32 + ph;
                f.push((rad * th.cos()) as f64); f.push((rad * th.sin()) as f64);
                t.push((rad * 0.995 * (th + 0.08).cos()) as f64); t.push((rad * 0.995 * (th + 0.08).sin()) as f64); }
            let o = m.step(&f); for q in 0..np { l += (o[2*q]-t[2*q]).powi(2) + (o[2*q+1]-t[2*q+1]).powi(2); c += 1; } }
        println!("TRAINLOSS {} MSE_1paso={:.6}", wpath, l / c as f64); return;
    }
    let seeds: Vec<u64> = match phase { "dev" => (0xA400..=0xA40F).collect(), "confirm" => (0xB400..=0xB40F).collect(), _ => panic!() };
    println!("pesos={} d_basis={} sinapsis={} (wp; wf=1e-4 fijo) | Dale viol (w<0)={} | w<1e-4={} | CONGELADO, sin adaptación", wpath, m.d, m.raw.len(), viol, below);
    let (mut s, mut e33p) = ([0.0f64; 4], 0);
    for &sd in &seeds {
        let (ka, kl, na, nl, p, e) = eval_seed(&m, sd, 64);
        println!("  Semilla {:#x}: known acc16={:5.1}% log={:7.3} | new acc16={:5.1}% log={:7.3} | E33={} {:?}", sd, ka * 100.0, kl, na * 100.0, nl, p, e);
        s[0] += ka; s[1] += kl; s[2] += na; s[3] += nl; if p { e33p += 1; }
    }
    let k = seeds.len() as f64;
    println!("RESUMEN {} {}: known acc16={:.2}% medlog={:.4} | new acc16={:.2}% medlog={:.4} | E33 {}/{} | Dale viol {}", phase.to_uppercase(), wpath, s[0] / k * 100.0, s[1] / k, s[2] / k * 100.0, s[3] / k, e33p, seeds.len(), viol);
}
