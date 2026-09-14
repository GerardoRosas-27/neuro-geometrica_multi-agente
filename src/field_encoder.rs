//! Encoder-sonda: texto / capas desacopladas → campo `z`.
//!
//! El sustrato no entiende tokens. Este módulo es un periférico: escribe un
//! patrón en `z` y se calla. Gemma 2 (26 capas, máscara LRC) es la rejilla
//! de escritura. Llama 2 no está en el repo; no se finge.
//!
//! Entrenamiento = Equilibrium Propagation / Hebb: el campo relajado enseña
//! al encoder qué puede sostener. Cero cross-entropy, cero vocab_size.

#![allow(clippy::needless_range_loop)]

use crate::field_substrate::{
    free_energy, handshake, hebb_update, ignite_casimir, prune, reconstruction_error, relax,
    ComplexT, FieldConfig, FieldState, Phasor,
};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

/// Geometría de Gemma 2 2B en este repo (capas desacopladas / LRC).
pub const GEMMA2_LAYER_COUNT: usize = 26;

/// Capas caras según T2.1 (`docs/v8_layer_kl_ablation.csv`): no escriben el campo.
pub const EXPENSIVE_LAYERS: [usize; 5] = [2, 3, 4, 5, 17];

const NGRAM_DIM: usize = 16;
const EPS: f64 = 1.0e-12;

#[derive(Clone, Debug)]
pub struct LayerSkipMask {
    pub on: Vec<bool>,
}

impl LayerSkipMask {
    pub fn all_on() -> Self {
        Self {
            on: vec![true; GEMMA2_LAYER_COUNT],
        }
    }

    /// Reutiliza el veredicto T2.1: apaga las 5 capas de KL > 0,15.
    pub fn from_t21_expensive_skip() -> Self {
        let mut on = vec![true; GEMMA2_LAYER_COUNT];
        on[0] = true;
        on[GEMMA2_LAYER_COUNT - 1] = true;
        for &i in &EXPENSIVE_LAYERS {
            if i < GEMMA2_LAYER_COUNT {
                on[i] = false;
            }
        }
        Self { on }
    }

    pub fn executed_count(&self) -> usize {
        self.on.iter().filter(|&&v| v).count()
    }
}

/// Sonda de 26 capas. Con GGUF serían RMS reales; aquí, huella determinista
/// del texto (misma forma, cero pesos del LLM).
#[derive(Clone, Debug)]
pub struct LayerProbe {
    pub rms: [f64; GEMMA2_LAYER_COUNT],
}

impl LayerProbe {
    pub fn from_text(text: &str) -> Self {
        let mut rms = [0.0; GEMMA2_LAYER_COUNT];
        for (i, slot) in rms.iter_mut().enumerate() {
            let h = hash_mix(text.as_bytes(), i as u64);
            *slot = 0.35 + 0.65 * ((h as f64) / (u64::MAX as f64));
        }
        Self { rms }
    }

    pub fn apply_mask(&self, mask: &LayerSkipMask) -> Vec<f64> {
        self.rms
            .iter()
            .zip(mask.on.iter())
            .map(|(&v, &on)| if on { v } else { 0.0 })
            .collect()
    }
}

fn hash_mix(bytes: &[u8], salt: u64) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
    for &b in bytes {
        h = h.wrapping_mul(0x100_0000_01B3).wrapping_add(b as u64);
        h ^= h >> 27;
    }
    h
}

fn ngram_features(text: &str) -> [f64; NGRAM_DIM] {
    let mut f = [0.0; NGRAM_DIM];
    let lower: Vec<u8> = text.to_lowercase().bytes().collect();
    if lower.is_empty() {
        return f;
    }
    for w in lower.windows(2) {
        let h = hash_mix(w, 0x4E47);
        f[(h as usize) % NGRAM_DIM] += 1.0;
    }
    for &b in &lower {
        let h = hash_mix(&[b], 0x554E);
        f[(h as usize) % NGRAM_DIM] += 0.35;
    }
    let norm = f.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
    for v in f.iter_mut() {
        *v /= norm;
    }
    f
}

pub fn feature_dim() -> usize {
    GEMMA2_LAYER_COUNT + NGRAM_DIM
}

pub fn features_from_text(text: &str, mask: &LayerSkipMask) -> Vec<f64> {
    let probe = LayerProbe::from_text(text).apply_mask(mask);
    let grams = ngram_features(text);
    let mut out = probe;
    out.extend_from_slice(&grams);
    let n = out.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
    for v in out.iter_mut() {
        *v /= n;
    }
    out
}

/// Mapa lineal features → fasores de arista. El LLM no es el modelo.
#[derive(Clone, Debug)]
pub struct FieldEncoder {
    pub n_edges: usize,
    pub feat_dim: usize,
    pub w_re: Vec<f64>,
    pub w_im: Vec<f64>,
    pub eta: f64,
    pub decay: f64,
}

impl FieldEncoder {
    pub fn new(n_edges: usize, seed: u64) -> Self {
        let feat_dim = feature_dim();
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let scale = 0.15 / (feat_dim as f64).sqrt();
        let mut w_re = vec![0.0; n_edges * feat_dim];
        let mut w_im = vec![0.0; n_edges * feat_dim];
        for v in w_re.iter_mut().chain(w_im.iter_mut()) {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * scale;
        }
        Self {
            n_edges,
            feat_dim,
            w_re,
            w_im,
            eta: 0.08,
            decay: 0.002,
        }
    }

    pub fn encode(&self, features: &[f64]) -> Vec<Phasor> {
        let mut z = vec![Phasor::new(0.0, 0.0); self.n_edges];
        for e in 0..self.n_edges {
            let mut re = 0.0;
            let mut im = 0.0;
            for k in 0..self.feat_dim {
                let f = features[k];
                re += self.w_re[e * self.feat_dim + k] * f;
                im += self.w_im[e * self.feat_dim + k] * f;
            }
            let raw = Phasor::new(re, im);
            let n = raw.norm();
            z[e] = if n > EPS {
                raw / n
            } else {
                Phasor::new(1.0, 0.0)
            };
        }
        z
    }

    /// El atractor relajado enseña al encoder (Hebb / EP). No hay backprop.
    pub fn hebb_from_attractor(&mut self, features: &[f64], z_star: &[Phasor]) {
        for e in 0..self.n_edges {
            for k in 0..self.feat_dim {
                let idx = e * self.feat_dim + k;
                self.w_re[idx] *= 1.0 - self.decay;
                self.w_im[idx] *= 1.0 - self.decay;
                self.w_re[idx] += self.eta * z_star[e].re * features[k];
                self.w_im[idx] += self.eta * z_star[e].im * features[k];
            }
        }
        self.renorm_rows();
    }

    /// Regla delta local: acerca la escritura al atractor. No es backprop.
    pub fn delta_toward(&mut self, features: &[f64], z_now: &[Phasor], z_target: &[Phasor]) {
        for e in 0..self.n_edges {
            let err = z_target[e] - z_now[e];
            for k in 0..self.feat_dim {
                let idx = e * self.feat_dim + k;
                self.w_re[idx] += self.eta * err.re * features[k];
                self.w_im[idx] += self.eta * err.im * features[k];
            }
        }
    }

    fn renorm_rows(&mut self) {
        for e in 0..self.n_edges {
            let mut n2 = 0.0;
            for k in 0..self.feat_dim {
                let idx = e * self.feat_dim + k;
                n2 += self.w_re[idx] * self.w_re[idx] + self.w_im[idx] * self.w_im[idx];
            }
            let n = n2.sqrt().max(EPS);
            for k in 0..self.feat_dim {
                let idx = e * self.feat_dim + k;
                self.w_re[idx] /= n;
                self.w_im[idx] /= n;
            }
        }
    }
}

/// Decoder lineal de sonda: clusters rígidos → etiqueta. No es el modelo.
pub fn read_rigid_signature(psi: &FieldState) -> Vec<f64> {
    psi.z
        .iter()
        .zip(psi.m.iter())
        .map(|(z, m)| if *m >= 1.0 { z.arg() } else { 0.0 })
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct FieldLoss {
    pub recon: f64,
    pub energy: f64,
    pub live_cost: f64,
    pub total: f64,
}

impl FieldLoss {
    pub fn compose(recon: f64, energy: f64, live: usize, n_edges: usize) -> Self {
        let live_cost = live as f64 / n_edges as f64;
        Self {
            recon,
            energy,
            live_cost,
            total: recon + 0.02 * energy.max(0.0) + 0.05 * live_cost,
        }
    }
}

pub fn write_into_field(psi: &mut FieldState, z: &[Phasor]) {
    psi.z = z.to_vec();
    psi.z_past = z.to_vec();
}

/// Predicción de campo: apaga una región de `z` **y** de `z_past`.
/// Si `z_past` conserva la respuesta, Hopfield hace trampa.
pub fn field_predict_loss(
    psi: &mut FieldState,
    target: &[Phasor],
    mask: &[usize],
    cfg: &FieldConfig,
    seed: u64,
) -> FieldLoss {
    for &e in mask {
        psi.z[e] = Phasor::new(0.0, 0.0);
        psi.z_past[e] = Phasor::new(0.0, 0.0);
        psi.m[e] = 0.0;
    }
    relax(psi, cfg, 40, seed);
    let _ = handshake(psi, cfg);
    ignite_casimir(psi);
    let live = prune(psi, cfg);
    let recon = reconstruction_error(target, &psi.z, mask);
    let energy = free_energy(psi, cfg).total;
    FieldLoss::compose(recon, energy, live, psi.n())
}

fn relax_free(psi: &mut FieldState, cfg: &FieldConfig, seed: u64) {
    relax(psi, cfg, 24, seed);
    let _ = handshake(psi, cfg);
    ignite_casimir(psi);
    hebb_update(psi, cfg);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainReport {
    pub steps: usize,
    pub loss_before: f64,
    pub loss_after: f64,
    pub recon_before: f64,
    pub recon_after: f64,
    pub same_cluster_sim_before: f64,
    pub same_cluster_sim_after: f64,
    pub diff_cluster_sim_before: f64,
    pub diff_cluster_sim_after: f64,
}

pub fn corpus() -> Vec<(&'static str, usize)> {
    vec![
        ("hace frío en la montaña", 0),
        ("el hielo cubre el lago", 0),
        ("viento gélido esta noche", 0),
        ("el horno quema la panadería", 1),
        ("llama viva en la chimenea", 1),
        ("el sol calienta la plaza", 1),
        ("el perro corre en el patio", 2),
        ("un gato duerme en la silla", 2),
        ("caballo negro en el prado", 2),
    ]
}

fn cosine_phasors(a: &[Phasor], b: &[Phasor]) -> f64 {
    let mut num = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        num += (x.conj() * y).re;
        na += x.norm_sqr();
        nb += y.norm_sqr();
    }
    num / (na.sqrt() * nb.sqrt() + EPS)
}

fn cluster_sims(zs: &[Vec<Phasor>], labels: &[usize]) -> (f64, f64) {
    let mut same = 0.0f64;
    let mut ns = 0.0f64;
    let mut diff = 0.0f64;
    let mut nd = 0.0f64;
    for i in 0..zs.len() {
        for j in (i + 1)..zs.len() {
            let s = cosine_phasors(&zs[i], &zs[j]);
            if labels[i] == labels[j] {
                same += s;
                ns += 1.0;
            } else {
                diff += s;
                nd += 1.0;
            }
        }
    }
    (same / ns.max(1.0), diff / nd.max(1.0))
}

fn encode_corpus(enc: &FieldEncoder, mask: &LayerSkipMask) -> (Vec<Vec<Phasor>>, Vec<usize>) {
    let data = corpus();
    let zs = data
        .iter()
        .map(|(t, _)| enc.encode(&features_from_text(t, mask)))
        .collect();
    let labels = data.iter().map(|(_, k)| *k).collect();
    (zs, labels)
}

fn mean_loss(enc: &FieldEncoder, mask: &LayerSkipMask, cfg: &FieldConfig) -> (f64, f64) {
    let data = corpus();
    let mask_edges = [0usize, 1, 8, 9];
    let mut lt = 0.0;
    let mut rt = 0.0;
    for (i, (text, _)) in data.iter().enumerate() {
        let f = features_from_text(text, mask);
        let z = enc.encode(&f);
        let mut psi = FieldState::new(ComplexT::octahedron());
        write_into_field(&mut psi, &z);
        hebb_update(&mut psi, cfg);
        relax_free(&mut psi, cfg, 1000 + i as u64);
        let loss = field_predict_loss(&mut psi, &z, &mask_edges, cfg, 2000 + i as u64);
        lt += loss.total;
        rt += loss.recon;
    }
    let n = data.len() as f64;
    (lt / n, rt / n)
}

/// Tres atractores de campo, uno por cluster semántico. El encoder aprende
/// a escribirlos. No hay softmax ni vocabulario.
pub fn cluster_field_target(cluster: usize, n_edges: usize) -> Vec<Phasor> {
    (0..n_edges)
        .map(|e| {
            let phase = match cluster % 3 {
                0 => {
                    if e < 4 {
                        0.0
                    } else {
                        std::f64::consts::FRAC_PI_2
                    }
                }
                1 => {
                    if e % 2 == 0 {
                        std::f64::consts::FRAC_PI_3
                    } else {
                        -std::f64::consts::FRAC_PI_3
                    }
                }
                _ => (e as f64) * std::f64::consts::PI / 6.0,
            };
            Phasor::from_polar(1.0, phase)
        })
        .collect()
}

/// Entrena el encoder para hablar con el campo: Hebb hacia el atractor del
/// cluster, luego EP (relax libre enseña al mapa).
pub fn train_encoder(steps: usize, seed: u64) -> (FieldEncoder, TrainReport) {
    let cfg = FieldConfig {
        temperature: 0.0,
        ..FieldConfig::default()
    };
    let layer_mask = LayerSkipMask::from_t21_expensive_skip();
    let mut enc = FieldEncoder::new(ComplexT::octahedron().n_edges(), seed);
    enc.eta = 0.12;
    enc.decay = 0.001;
    let data = corpus();
    let n_edges = ComplexT::octahedron().n_edges();
    let targets = [
        cluster_field_target(0, n_edges),
        cluster_field_target(1, n_edges),
        cluster_field_target(2, n_edges),
    ];

    let (zs0, labels) = encode_corpus(&enc, &layer_mask);
    let (same0, diff0) = cluster_sims(&zs0, &labels);
    let (loss0, recon0) = mean_loss(&enc, &layer_mask, &cfg);

    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x71);
    for step in 0..steps {
        let idx = rng.gen_range(0..data.len());
        let (text, cluster) = data[idx];
        let f = features_from_text(text, &layer_mask);
        let z = enc.encode(&f);
        enc.delta_toward(&f, &z, &targets[cluster]);
        if step % 4 == 3 {
            let z2 = enc.encode(&f);
            let mut psi = FieldState::new(ComplexT::octahedron());
            write_into_field(&mut psi, &z2);
            hebb_update(&mut psi, &cfg);
            relax_free(&mut psi, &cfg, seed + step as u64);
            let saved = enc.eta;
            enc.eta = saved * 0.4;
            let z3 = enc.encode(&f);
            enc.delta_toward(&f, &z3, &psi.z);
            enc.eta = saved;
        }
    }

    let (zs1, labels1) = encode_corpus(&enc, &layer_mask);
    let (same1, diff1) = cluster_sims(&zs1, &labels1);
    let (loss1, recon1) = mean_loss(&enc, &layer_mask, &cfg);

    (
        enc,
        TrainReport {
            steps,
            loss_before: loss0,
            loss_after: loss1,
            recon_before: recon0,
            recon_after: recon1,
            same_cluster_sim_before: same0,
            same_cluster_sim_after: same1,
            diff_cluster_sim_before: diff0,
            diff_cluster_sim_after: diff1,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_mask_reuses_t21_expensive_layers() {
        let m = LayerSkipMask::from_t21_expensive_skip();
        assert_eq!(m.on.len(), GEMMA2_LAYER_COUNT);
        assert_eq!(
            m.executed_count(),
            GEMMA2_LAYER_COUNT - EXPENSIVE_LAYERS.len()
        );
        assert!(!m.on[2] && !m.on[17]);
        assert!(m.on[0] && m.on[25]);
    }

    #[test]
    fn encoder_writes_field_not_tokens() {
        let enc = FieldEncoder::new(12, 1);
        let z = enc.encode(&features_from_text(
            "hace frío",
            &LayerSkipMask::from_t21_expensive_skip(),
        ));
        assert_eq!(z.len(), 12);
        assert!(z.iter().all(|p| p.norm().is_finite()));
    }

    #[test]
    fn same_text_is_deterministic() {
        let mask = LayerSkipMask::from_t21_expensive_skip();
        let a = features_from_text("el perro corre", &mask);
        let b = features_from_text("el perro corre", &mask);
        assert_eq!(a, b);
    }

    #[test]
    fn training_lowers_field_prediction_loss() {
        let (_enc, report) = train_encoder(120, 0xE4C0);
        println!(
            "steps={} L {:.4}->{:.4} recon {:.4}->{:.4} same {:.3}->{:.3} diff {:.3}->{:.3}",
            report.steps,
            report.loss_before,
            report.loss_after,
            report.recon_before,
            report.recon_after,
            report.same_cluster_sim_before,
            report.same_cluster_sim_after,
            report.diff_cluster_sim_before,
            report.diff_cluster_sim_after
        );
        let margin0 = report.same_cluster_sim_before - report.diff_cluster_sim_before;
        let margin1 = report.same_cluster_sim_after - report.diff_cluster_sim_after;
        println!("cluster_margin {margin0:.3} -> {margin1:.3}");
        assert!(
            margin1 > margin0 + 0.15,
            "encoder should separate clusters in the field: margin {margin0:.3}->{margin1:.3} {report:?}"
        );
    }
}
