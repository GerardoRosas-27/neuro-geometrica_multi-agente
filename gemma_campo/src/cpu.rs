//! Referencia CPU (f64 donde importa): pérdida, paso SGD idéntico al kernel, ridge M7 por grado,
//! campo rápido NLMS y rollouts multi-paso. Todo lo que hace la GPU tiene aquí su gemelo.
use crate::data::Split;
use crate::features::FeatCfg;
use serde::{Deserialize, Serialize};

/// Lista de muestras (conv, t) con objetivo z_{t+1}.
pub fn samples(s: &Split) -> Vec<(u32, u32)> {
    let mut v = Vec::new();
    for (c, &l) in s.lengths.iter().enumerate() { for t in 0..l.saturating_sub(1) { v.push((c as u32, t as u32)); } }
    v
}

/// Construye phi (n x F) e y = z_{t+1}-z_t (n x d) para un lote de muestras.
pub fn build_batch(cfg: &FeatCfg, s: &Split, idx: &[(u32, u32)], phi: &mut [f32], y: &mut [f32]) {
    let (f, d) = (cfg.dim(), cfg.d);
    for (b, &(c, t)) in idx.iter().enumerate() {
        let seq = s.conv(c as usize);
        let t = t as usize;
        cfg.build(seq, t, &mut phi[b * f..(b + 1) * f]);
        for o in 0..d { y[b * d + o] = seq[(t + 1) * d + o] - seq[t * d + o]; }
    }
}

/// MSE de datos (media sobre n*d), acumulado en f64.
pub fn loss_ref(phi: &[f32], y: &[f32], w: &[f32], n: usize, f: usize, d: usize) -> f64 {
    let mut s = 0.0f64;
    for b in 0..n { for o in 0..d {
        let mut acc = 0.0f64;
        for j in 0..f { acc += phi[b * f + j] as f64 * w[j * d + o] as f64; }
        let e = acc - y[b * d + o] as f64; s += e * e;
    } }
    s / (n * d) as f64
}

/// Gemelo CPU exacto de k_forward + k_update (mismo orden lógico: err con W antiguo, luego actualización).
pub fn sgd_step_ref(phi: &[f32], y: &[f32], w: &mut [f32], v: &mut [f32], lam: &[f32],
                    n: usize, f: usize, d: usize, lr: f32, mom: f32) -> f64 {
    let mut err = vec![0f64; n * d];
    let mut s = 0.0;
    for b in 0..n { for o in 0..d {
        let mut acc = 0.0f64;
        for j in 0..f { acc += phi[b * f + j] as f64 * w[j * d + o] as f64; }
        let e = acc - y[b * d + o] as f64; err[b * d + o] = e; s += e * e;
    } }
    for j in 0..f { for o in 0..d {
        let mut g = 0.0f64;
        for b in 0..n { g += phi[b * f + j] as f64 * err[b * d + o]; }
        let i = j * d + o;
        let g = g * (2.0 / (n * d) as f64) + 2.0 * lam[j] as f64 * w[i] as f64;
        let vn = mom as f64 * v[i] as f64 + g;
        v[i] = vn as f32;
        w[i] = (w[i] as f64 - lr as f64 * vn) as f32;
    } }
    s / (n * d) as f64
}

/// Estadísticos suficientes: G = Phi^T Phi (F x F), B = Phi^T Y (F x d), yy = sum y^2, n.
pub struct Stats { pub f: usize, pub d: usize, pub g: Vec<f64>, pub b: Vec<f64>, pub yy: f64, pub n: usize }

pub fn gram(cfg: &FeatCfg, s: &Split, idx: &[(u32, u32)], threads: usize) -> Stats {
    let (f, d) = (cfg.dim(), cfg.d);
    let chunks: Vec<&[(u32, u32)]> = idx.chunks(idx.len().div_ceil(threads.max(1)).max(1)).collect();
    let parts: Vec<Stats> = std::thread::scope(|sc| {
        let hs: Vec<_> = chunks.iter().map(|ch| sc.spawn(move || {
            let mut st = Stats { f, d, g: vec![0.0; f * f], b: vec![0.0; f * d], yy: 0.0, n: 0 };
            let mut phi = vec![0f32; f]; let mut y = vec![0f32; d];
            let mut p64 = vec![0f64; f];
            for &sm in ch.iter() {
                build_batch(cfg, s, &[sm], &mut phi, &mut y);
                for j in 0..f { p64[j] = phi[j] as f64; }
                for i in 0..f { let pi = p64[i]; if pi == 0.0 { continue; }
                    let row = &mut st.g[i * f..(i + 1) * f];
                    for j in i..f { row[j] += pi * p64[j]; }
                    for o in 0..d { st.b[i * d + o] += pi * y[o] as f64; } }
                for o in 0..d { st.yy += (y[o] as f64).powi(2); }
                st.n += 1;
            }
            st
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut tot = Stats { f, d, g: vec![0.0; f * f], b: vec![0.0; f * d], yy: 0.0, n: 0 };
    for p in parts { for (a, x) in tot.g.iter_mut().zip(&p.g) { *a += x; } for (a, x) in tot.b.iter_mut().zip(&p.b) { *a += x; } tot.yy += p.yy; tot.n += p.n; }
    for i in 0..f { for j in 0..i { tot.g[i * f + j] = tot.g[j * f + i]; } }
    tot
}

/// Resuelve (G + n*d*diag(lam)) W = B por Cholesky. Mismo objetivo que el SGD (L definido en train.wgsl).
pub fn ridge_solve(st: &Stats, lam: &[f64]) -> Option<Vec<f64>> {
    let (f, d) = (st.f, st.d);
    let mut a = st.g.clone();
    let sc = (st.n * d) as f64;
    for i in 0..f { a[i * f + i] += sc * lam[i] + 1e-9 * (1.0 + st.g[i * f + i]); }
    // Cholesky in-place (triangular inferior)
    for j in 0..f {
        let mut s = a[j * f + j];
        for k in 0..j { s -= a[j * f + k] * a[j * f + k]; }
        if s <= 0.0 { return None; }
        let l = s.sqrt(); a[j * f + j] = l;
        for i in j + 1..f {
            let mut s = a[i * f + j];
            let (ri, rj) = (i * f, j * f);
            for k in 0..j { s -= a[ri + k] * a[rj + k]; }
            a[i * f + j] = s / l;
        }
    }
    let mut w = st.b.clone();
    for o in 0..d {
        for i in 0..f { let mut s = w[i * d + o]; for k in 0..i { s -= a[i * f + k] * w[k * d + o]; } w[i * d + o] = s / a[i * f + i]; }
        for i in (0..f).rev() { let mut s = w[i * d + o]; for k in i + 1..f { s -= a[k * f + i] * w[k * d + o]; } w[i * d + o] = s / a[i * f + i]; }
    }
    Some(w)
}

/// MSE de datos usando estadísticos: (yy - 2 tr(W^T B) + tr(W^T G W)) / (n d)
pub fn mse_from_stats(st: &Stats, w: &[f64]) -> f64 {
    let (f, d) = (st.f, st.d);
    let mut s = st.yy;
    for i in 0..f * d { s -= 2.0 * w[i] * st.b[i]; }
    let mut gw = vec![0f64; f * d];
    for i in 0..f { for k in 0..f { let g = st.g[i * f + k]; if g == 0.0 { continue; } for o in 0..d { gw[i * d + o] += g * w[k * d + o]; } } }
    for i in 0..f * d { s += w[i] * gw[i]; }
    s / (st.n * d) as f64
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelMeta {
    pub cfg: FeatCfg,
    pub lam_deg: [f64; 3],
    pub fast_eta: f64,
    pub fast_gamma: f64,
    pub note: String,
}

pub struct Model { pub meta: ModelMeta, pub w: Vec<f32> }

/// Campo rápido: A (d x d) por conversación.
pub struct Fast { pub a: Vec<f64>, pub d: usize, pub eta: f64, pub gamma: f64 }
impl Fast {
    pub fn new(d: usize, eta: f64, gamma: f64) -> Self { Fast { a: vec![0.0; d * d], d, eta, gamma } }
    pub fn apply(&self, z: &[f32], out: &mut [f64]) {
        if self.eta == 0.0 { return; }
        let d = self.d;
        for i in 0..d { let mut s = 0.0; for j in 0..d { s += self.a[i * d + j] * z[j] as f64; } out[i] += s; }
    }
    /// NLMS: r = residuo de la lenta (z_{t+1} - slow_pred); e = r - A z
    pub fn update(&mut self, z: &[f32], r: &[f64]) {
        if self.eta == 0.0 { return; }
        let d = self.d;
        let nz: f64 = z.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() + 1e-6;
        let mut az = vec![0f64; d];
        for i in 0..d { for j in 0..d { az[i] += self.a[i * d + j] * z[j] as f64; } }
        for i in 0..d { let e = r[i] - az[i]; for j in 0..d {
            let k = i * d + j; self.a[k] = self.gamma * self.a[k] + self.eta * e * z[j] as f64 / nz; } }
    }
}

impl Model {
    /// Predicción lenta de z_{t+1} dado seq[..=t]
    pub fn slow(&self, seq: &[f32], t: usize, phi: &mut [f32], out: &mut [f64]) {
        let c = &self.meta.cfg; let (f, d) = (c.dim(), c.d);
        c.build(seq, t, phi);
        for o in 0..d { out[o] = seq[t * d + o] as f64; }
        for j in 0..f { let p = phi[j] as f64; if p == 0.0 { continue; } for o in 0..d { out[o] += p * self.w[j * d + o] as f64; } }
    }
}

/// Resultado por conversación: MSE medio en z por método y horizonte.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ConvEval { pub conv: usize, pub n1: usize, pub field1: f64, pub slow1: f64, pub repeat1: f64,
    pub nk: Vec<usize>, pub fieldk: Vec<f64>, pub slowk: Vec<f64>, pub repeatk: Vec<f64> }

/// Evalúa una conversación. Si `pred_out` se da, escribe ahí la predicción 1-paso del campo (fila t = pred de z_t,
/// fila 0 = z_0 sin predicción; se excluye en métricas).
pub fn eval_conv(m: &Model, seq: &[f32], len: usize, kmax: usize, stride: usize, eta: f64, gamma: f64,
                 mut pred_out: Option<&mut [f32]>) -> ConvEval {
    let d = m.meta.cfg.d; let f = m.meta.cfg.dim();
    let mut phi = vec![0f32; f]; let mut sl = vec![0f64; d];
    let mut fast = Fast::new(d, eta, gamma);
    let mut r = ConvEval { nk: vec![0; kmax], fieldk: vec![0.0; kmax], slowk: vec![0.0; kmax], repeatk: vec![0.0; kmax], ..Default::default() };
    if let Some(p) = pred_out.as_deref_mut() { p[..d].copy_from_slice(&seq[..d]); }
    for t in 0..len.saturating_sub(1) {
        // rollout multi-paso desde t (A congelada)
        if kmax > 1 && t % stride == 0 {
            for (mode, acc) in [(0usize, &mut r.fieldk), (1, &mut r.slowk)] {
                let mut ext: Vec<f32> = seq[..(t + 1) * d].to_vec();
                for k in 1..=kmax.min(len - 1 - t) {
                    let tc = t + k - 1;
                    m.slow(&ext, tc, &mut phi, &mut sl);
                    if mode == 0 { fast.apply(&ext[tc * d..(tc + 1) * d], &mut sl); }
                    let tgt = &seq[(t + k) * d..(t + k + 1) * d];
                    acc[k - 1] += (0..d).map(|o| (sl[o] - tgt[o] as f64).powi(2)).sum::<f64>() / d as f64;
                    ext.extend(sl.iter().map(|x| *x as f32));
                }
            }
            for k in 1..=kmax.min(len - 1 - t) {
                let tgt = &seq[(t + k) * d..(t + k + 1) * d];
                r.repeatk[k - 1] += (0..d).map(|o| ((seq[t * d + o] - tgt[o]) as f64).powi(2)).sum::<f64>() / d as f64;
                r.nk[k - 1] += 1;
            }
        }
        // 1 paso
        m.slow(seq, t, &mut phi, &mut sl);
        let tgt = &seq[(t + 1) * d..(t + 2) * d];
        let zt = &seq[t * d..(t + 1) * d];
        let mut fp = sl.clone(); fast.apply(zt, &mut fp);
        r.slow1 += (0..d).map(|o| (sl[o] - tgt[o] as f64).powi(2)).sum::<f64>() / d as f64;
        r.field1 += (0..d).map(|o| (fp[o] - tgt[o] as f64).powi(2)).sum::<f64>() / d as f64;
        r.repeat1 += (0..d).map(|o| ((zt[o] - tgt[o]) as f64).powi(2)).sum::<f64>() / d as f64;
        r.n1 += 1;
        if let Some(p) = pred_out.as_deref_mut() { for o in 0..d { p[(t + 1) * d + o] = fp[o] as f32; } }
        let res: Vec<f64> = (0..d).map(|o| tgt[o] as f64 - sl[o]).collect();
        fast.update(zt, &res);
    }
    let n1 = r.n1.max(1) as f64; r.field1 /= n1; r.slow1 /= n1; r.repeat1 /= n1;
    for k in 0..kmax { let n = r.nk[k].max(1) as f64; r.fieldk[k] /= n; r.slowk[k] /= n; r.repeatk[k] /= n; }
    r
}

/// Media sobre conversaciones (ponderada por nº de predicciones) de field1/slow1/repeat1 en 1 paso.
pub fn eval_split_1step(m: &Model, s: &Split, eta: f64, gamma: f64, threads: usize) -> (f64, f64, f64) {
    let convs: Vec<usize> = (0..s.lengths.len()).collect();
    let ch: Vec<&[usize]> = convs.chunks(convs.len().div_ceil(threads.max(1)).max(1)).collect();
    let rs: Vec<ConvEval> = std::thread::scope(|sc| {
        let hs: Vec<_> = ch.iter().map(|c| sc.spawn(move || c.iter().map(|&ci| { let mut e = eval_conv(m, s.conv(ci), s.lengths[ci], 1, 1, eta, gamma, None); e.conv = ci; e }).collect::<Vec<_>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let n: f64 = rs.iter().map(|r| r.n1 as f64).sum::<f64>().max(1.0);
    (rs.iter().map(|r| r.field1 * r.n1 as f64).sum::<f64>() / n,
     rs.iter().map(|r| r.slow1 * r.n1 as f64).sum::<f64>() / n,
     rs.iter().map(|r| r.repeat1 * r.n1 as f64).sum::<f64>() / n)
}

/// lambda por feature a partir de lambda por grado
pub fn lam_per_feature(cfg: &FeatCfg, lam_deg: [f64; 3]) -> Vec<f64> {
    cfg.degree().iter().map(|&g| lam_deg[g as usize]).collect()
}

/// RNG determinista (splitmix64) para barajar / tests.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 { self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) }
    pub fn unif(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn normal(&mut self) -> f64 { let u = self.unif().max(1e-300); let v = self.unif(); (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos() }
    pub fn shuffle<T>(&mut self, v: &mut [T]) { for i in (1..v.len()).rev() { let j = (self.next() % (i as u64 + 1)) as usize; v.swap(i, j); } }
}
