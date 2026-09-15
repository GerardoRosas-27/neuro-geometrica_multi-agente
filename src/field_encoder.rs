//! Encoder-sonda: Gemma congelada (o huella 26-d) → campo `z`.
//!
//! Los tokens se quedan en esta capa lingüística. Al sustrato solo llega `z`.
//! Gemma 2 está congelada: no hay gradiente, no hay next-token, no hay Llama 2.
//!
//! Entrenamiento = regla delta hacia el atractor del cluster + EP ocasional.
//! El decoder rígido lee el banco de 3 clusters desde `z`.

#![allow(clippy::needless_range_loop)]

use crate::field_corpus::{self, example_at, is_holdout, sample_indices, N_TOPICS};
use crate::field_substrate::{
    commit_pattern, eval_mask, free_energy, handshake, ignite_cavities,
    mask_hidden, prune_amplitude, reconstruction_error, relax, write_hemisphere_pattern,
    write_meridian_pattern, write_pole_equator_pattern, ComplexT, FieldConfig, FieldState, Phasor,
};
use crate::native_checkpoint::atomic_write;
use crate::native_gemma2::{
    resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer, LayerExecutionMask,
    QuantizedGemma2,
};
use candle_core::quantized::gguf_file;
use candle_core::{IndexOp, Tensor};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Geometría de Gemma 2 2B en este repo (capas desacopladas / LRC).
pub const GEMMA2_LAYER_COUNT: usize = 26;

/// Capas caras según T2.1 (`docs/v8_layer_kl_ablation.csv`): no escriben el campo.
pub const EXPENSIVE_LAYERS: [usize; 5] = [2, 3, 4, 5, 17];

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

    pub fn execution_mask(&self, layer_count: usize) -> LayerExecutionMask {
        let mut enabled = vec![false; layer_count.max(1)];
        for (i, slot) in enabled.iter_mut().enumerate() {
            *slot = self.on.get(i).copied().unwrap_or(false);
        }
        LayerExecutionMask::from_enabled(enabled)
    }
}

/// Sonda de 26 capas. Misma forma que los RMS de Gemma; sin GGUF es huella.
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

    pub fn from_layer_rms(values: &[f32]) -> Self {
        let mut rms = [0.0; GEMMA2_LAYER_COUNT];
        for (i, slot) in rms.iter_mut().enumerate() {
            *slot = values.get(i).copied().unwrap_or(0.0) as f64;
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
        h = h.wrapping_mul(0x100_0000_01B3).wrapping_add(u64::from(b));
        h ^= h >> 27;
    }
    h
}

pub fn feature_dim() -> usize {
    GEMMA2_LAYER_COUNT
}

fn l2_normalize(mut out: Vec<f64>) -> Vec<f64> {
    let n = out.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
    for v in out.iter_mut() {
        *v /= n;
    }
    out
}

pub fn features_from_text(text: &str, mask: &LayerSkipMask) -> Vec<f64> {
    l2_normalize(LayerProbe::from_text(text).apply_mask(mask))
}

/// Gemma 2 congelada: tokeniza, corre el GGUF, entrega RMS por capa. Cero gradiente.
pub struct FrozenGemmaProbe {
    model: QuantizedGemma2,
    tokenizer: Gemma2Tokenizer,
    skip: LayerSkipMask,
    cache: HashMap<String, Vec<f64>>,
    pub model_path: PathBuf,
}

impl FrozenGemmaProbe {
    pub fn try_open() -> Result<Self, String> {
        let path = resolve_gemma2_model_path(None).map_err(|e| e.to_string())?;
        let device = resolve_gemma2_device("cpu").map_err(|e| e.to_string())?;
        let mut file = File::open(&path).map_err(|e| e.to_string())?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| e.to_string())?;
        let tokenizer = Gemma2Tokenizer::from_gguf(&content).map_err(|e| e.to_string())?;
        let model =
            QuantizedGemma2::from_gguf(content, &mut file, &device).map_err(|e| e.to_string())?;
        Ok(Self {
            model,
            tokenizer,
            skip: LayerSkipMask::from_t21_expensive_skip(),
            cache: HashMap::new(),
            model_path: path,
        })
    }

    pub fn hidden_dim(&self) -> usize {
        self.model.embedding_length()
    }

    pub fn features(&mut self, text: &str) -> Result<Vec<f64>, String> {
        if let Some(hit) = self.cache.get(text) {
            return Ok(hit.clone());
        }
        let mut tokens = vec![self.tokenizer.bos_id];
        tokens.extend(self.tokenizer.encode(text).map_err(|e| e.to_string())?);
        if tokens.len() == 1 {
            return Err("texto vacío tras tokenizar".into());
        }
        self.model.clear_kv_cache();
        let ids = Tensor::new(tokens.as_slice(), self.model.device())
            .map_err(|e| e.to_string())?
            .unsqueeze(0)
            .map_err(|e| e.to_string())?;
        let exec = self.skip.execution_mask(self.model.layer_count());
        let output = self
            .model
            .forward_with_mask(&ids, 0, Some(&exec), None, true, true)
            .map_err(|e| e.to_string())?;
        let hidden = output
            .sequence_hidden
            .ok_or_else(|| "Gemma no devolvió hidden".to_string())?;
        let seq = hidden.dim(1).map_err(|e| e.to_string())?;
        let last = hidden
            .i((0, seq.saturating_sub(1), ..))
            .map_err(|e| e.to_string())?
            .to_vec1::<f32>()
            .map_err(|e| e.to_string())?;
        let feats = l2_normalize(last.into_iter().map(f64::from).collect());
        if self.cache.len() >= 1024 {
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(text.to_string(), feats.clone());
        Ok(feats)
    }
}

/// Capa lingüística del encoder. Los tokens no salen de aquí.
pub enum LinguisticLayer {
    Hash { mask: LayerSkipMask },
    FrozenGemma { probe: FrozenGemmaProbe },
}

impl LinguisticLayer {
    pub fn open(prefer_frozen_gemma: bool) -> Self {
        if prefer_frozen_gemma {
            match FrozenGemmaProbe::try_open() {
                Ok(probe) => return Self::FrozenGemma { probe },
                Err(_) => {}
            }
        }
        Self::Hash {
            mask: LayerSkipMask::from_t21_expensive_skip(),
        }
    }

    pub fn source(&self) -> &'static str {
        match self {
            Self::Hash { .. } => "hash-probe-fallback",
            Self::FrozenGemma { .. } => "frozen-gemma2-gguf",
        }
    }

    pub fn feature_dim(&self) -> usize {
        match self {
            Self::Hash { .. } => GEMMA2_LAYER_COUNT,
            Self::FrozenGemma { probe } => probe.hidden_dim(),
        }
    }

    pub fn features(&mut self, text: &str) -> Result<Vec<f64>, String> {
        match self {
            Self::Hash { mask } => Ok(features_from_text(text, mask)),
            Self::FrozenGemma { probe } => probe.features(text),
        }
    }
}

/// Mapa lineal features → fasores de arista. El LLM no es el modelo.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
        Self::with_feat_dim(n_edges, feature_dim(), seed)
    }

    pub fn with_feat_dim(n_edges: usize, feat_dim: usize, seed: u64) -> Self {
        let feat_dim = feat_dim.max(1);
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
            for k in 0..self.feat_dim.min(features.len()) {
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

    pub fn hebb_from_attractor(&mut self, features: &[f64], z_star: &[Phasor]) {
        for e in 0..self.n_edges {
            for k in 0..self.feat_dim.min(features.len()) {
                let idx = e * self.feat_dim + k;
                self.w_re[idx] *= 1.0 - self.decay;
                self.w_im[idx] *= 1.0 - self.decay;
                self.w_re[idx] += self.eta * z_star[e].re * features[k];
                self.w_im[idx] += self.eta * z_star[e].im * features[k];
            }
        }
        self.renorm_rows();
    }

    pub fn delta_toward(&mut self, features: &[f64], z_now: &[Phasor], z_target: &[Phasor]) {
        for e in 0..self.n_edges.min(z_now.len()).min(z_target.len()) {
            let err = z_target[e] - z_now[e];
            for k in 0..self.feat_dim.min(features.len()) {
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

/// Decoder rígido: `z` → banco de 3 clusters. No es un modelo de palabras.
pub fn decode_cluster_bank(z: &[Phasor]) -> usize {
    let targets = [
        cluster_field_target(0),
        cluster_field_target(1),
        cluster_field_target(2),
    ];
    decode_against(z, &targets)
}

pub fn decoder_accuracy(
    enc: &FieldEncoder,
    ling: &mut LinguisticLayer,
    data: &[(&str, usize)],
) -> Result<f64, String> {
    let targets = [
        cluster_field_target(0),
        cluster_field_target(1),
        cluster_field_target(2),
    ];
    decoder_accuracy_with(enc, ling, data, &targets)
}

pub fn decoder_accuracy_with(
    enc: &FieldEncoder,
    ling: &mut LinguisticLayer,
    data: &[(&str, usize)],
    targets: &[Vec<Phasor>],
) -> Result<f64, String> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let mut ok = 0.0;
    for (text, cluster) in data {
        let z = enc.encode(&ling.features(text)?);
        if decode_against(&z, targets) == *cluster {
            ok += 1.0;
        }
    }
    Ok(ok / data.len() as f64)
}

/// Decoder lineal de sonda: fases rígidas. No es el modelo.
pub fn read_rigid_signature(psi: &FieldState) -> Vec<f64> {
    psi.z
        .iter()
        .zip(psi.m.iter())
        .map(|(z, m)| if *m >= 1.0 { z.arg() } else { 0.0 })
        .collect()
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct FieldLoss {
    pub recon: f64,
    pub energy: f64,
    pub live_cost: f64,
    pub total: f64,
}

impl FieldLoss {
    pub fn compose(recon: f64, energy: f64, live: usize, n_edges: usize) -> Self {
        let live_cost = live as f64 / n_edges.max(1) as f64;
        Self {
            recon,
            energy,
            live_cost,
            total: recon + 0.02 * energy.max(0.0) + 0.05 * live_cost,
        }
    }
}

pub fn write_into_field(psi: &mut FieldState, z: &[Phasor]) {
    let n = psi.n().min(z.len());
    psi.z[..n].copy_from_slice(&z[..n]);
}

/// Predicción de campo: apaga una región de `z` **y** de `z_past`.
pub fn field_predict_loss(
    psi: &mut FieldState,
    target: &[Phasor],
    mask: &[usize],
    cfg: &FieldConfig,
    seed: u64,
) -> FieldLoss {
    mask_hidden(psi, mask);
    ignite_cavities(psi);
    relax(psi, cfg, 40, seed);
    let _ = handshake(psi, cfg);
    ignite_cavities(psi);
    let live = prune_amplitude(psi, cfg);
    let recon = reconstruction_error(target, &psi.z, mask);
    let energy = free_energy(psi, cfg).total;
    FieldLoss::compose(recon, energy, live, psi.n())
}

fn relax_free(psi: &mut FieldState, cfg: &FieldConfig, seed: u64) {
    relax(psi, cfg, 24, seed);
    let _ = handshake(psi, cfg);
    ignite_cavities(psi);
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainReport {
    pub steps: usize,
    pub steps_done: usize,
    pub linguistic_source: String,
    pub loss_before: f64,
    pub loss_after: f64,
    pub recon_before: f64,
    pub recon_after: f64,
    pub same_cluster_sim_before: f64,
    pub same_cluster_sim_after: f64,
    pub diff_cluster_sim_before: f64,
    pub diff_cluster_sim_after: f64,
    pub decoder_acc_before: f64,
    pub decoder_acc_after: f64,
    pub decoder_holdout_before: f64,
    pub decoder_holdout_after: f64,
    #[serde(default)]
    pub n_edges: usize,
    #[serde(default)]
    pub n_clusters: usize,
    #[serde(default)]
    pub corpus_size: usize,
    #[serde(default)]
    pub geometry_level: usize,
    #[serde(default)]
    pub hours_saved: u32,
}

impl TrainReport {
    pub fn margin_before(&self) -> f64 {
        self.same_cluster_sim_before - self.diff_cluster_sim_before
    }

    pub fn margin_after(&self) -> f64 {
        self.same_cluster_sim_after - self.diff_cluster_sim_after
    }
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

pub fn holdout_corpus() -> Vec<(&'static str, usize)> {
    vec![
        ("nieve sobre el pico", 0),
        ("brasas en la estufa", 1),
        ("la yegua pasta quieta", 2),
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

fn encode_items(
    enc: &FieldEncoder,
    ling: &mut LinguisticLayer,
    data: &[(&str, usize)],
) -> Result<(Vec<Vec<Phasor>>, Vec<usize>), String> {
    let mut zs = Vec::with_capacity(data.len());
    let mut labels = Vec::with_capacity(data.len());
    for (text, cluster) in data {
        zs.push(enc.encode(&ling.features(text)?));
        labels.push(*cluster);
    }
    Ok((zs, labels))
}

/// Tres atractores geométricos, uno por cluster. 6+6 / meridiano / polos-ecuador.
pub fn cluster_field_target(cluster: usize) -> Vec<Phasor> {
    cluster_target_on(&ComplexT::octahedron(), cluster, 3)
}

pub fn cluster_target_on(t: &ComplexT, cluster: usize, n_clusters: usize) -> Vec<Phasor> {
    if n_clusters == 3 && t.n_edges() == 12 {
        let mut psi = FieldState::new(t.clone());
        match cluster % 3 {
            0 => write_hemisphere_pattern(&mut psi),
            1 => write_meridian_pattern(&mut psi),
            _ => write_pole_equator_pattern(&mut psi),
        }
        return psi.z;
    }
    let k = n_clusters.max(1);
    let n = t.n_edges();
    let base = std::f64::consts::TAU * (cluster % k) as f64 / k as f64;
    (0..n)
        .map(|e| {
            let flip = (e + cluster * 5) % 2 == 0;
            Phasor::from_polar(
                1.0,
                if flip {
                    base
                } else {
                    base + std::f64::consts::PI
                },
            )
        })
        .collect()
}

pub fn decode_against(z: &[Phasor], targets: &[Vec<Phasor>]) -> usize {
    let mut best = 0usize;
    let mut best_s = f64::NEG_INFINITY;
    for (k, target) in targets.iter().enumerate() {
        let s = cosine_phasors(z, target);
        if s > best_s {
            best_s = s;
            best = k;
        }
    }
    best
}

#[derive(Clone, Debug)]
pub struct EncoderTrainConfig {
    pub steps: usize,
    pub seed: u64,
    pub checkpoint_dir: Option<PathBuf>,
    pub checkpoint_every: usize,
    pub resume: bool,
    pub prefer_frozen_gemma: bool,
    pub geometry_level: usize,
    pub corpus_size: usize,
    pub n_clusters: usize,
    pub checkpoint_every_secs: u64,
    pub max_hours: f64,
    pub eval_every: usize,
    pub log: bool,
    pub stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for EncoderTrainConfig {
    fn default() -> Self {
        Self {
            steps: 120,
            seed: 0xE4C0,
            checkpoint_dir: None,
            checkpoint_every: 30,
            resume: true,
            prefer_frozen_gemma: false,
            geometry_level: 0,
            corpus_size: 0,
            n_clusters: 3,
            checkpoint_every_secs: 0,
            max_hours: 0.0,
            eval_every: 0,
            log: false,
            stop: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderCheckpoint {
    pub version: u32,
    pub seed: u64,
    pub steps_total: usize,
    pub steps_done: usize,
    pub encoder: FieldEncoder,
    pub report: TrainReport,
    #[serde(default)]
    pub geometry_level: usize,
    #[serde(default)]
    pub corpus_size: usize,
    #[serde(default)]
    pub n_clusters: usize,
}

impl EncoderCheckpoint {
    pub const VERSION: u32 = 4;
}

pub fn save_encoder_checkpoint(path: &Path, ckpt: &EncoderCheckpoint) -> Result<(), String> {
    let body = serde_json::to_vec_pretty(ckpt).map_err(|e| e.to_string())?;
    atomic_write(path, &body)
}

pub fn load_encoder_checkpoint(path: &Path) -> Result<EncoderCheckpoint, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

pub fn train_encoder(steps: usize, seed: u64) -> (FieldEncoder, TrainReport) {
    train_encoder_with(EncoderTrainConfig {
        steps,
        seed,
        checkpoint_dir: None,
        resume: false,
        prefer_frozen_gemma: false,
        ..EncoderTrainConfig::default()
    })
    .expect("encoder train in-memory")
}

pub fn train_encoder_with(cfg: EncoderTrainConfig) -> Result<(FieldEncoder, TrainReport), String> {
    let field_cfg = FieldConfig {
        temperature: 0.0,
        ..FieldConfig::default()
    };
    let mut ling = LinguisticLayer::open(cfg.prefer_frozen_gemma);
    let t = ComplexT::scaled(cfg.geometry_level);
    let n_clusters = if cfg.corpus_size > 0 {
        cfg.n_clusters.max(1).min(N_TOPICS)
    } else {
        3
    };
    let stream = cfg.corpus_size > 0;
    let pool = if stream {
        cfg.corpus_size.max(n_clusters * 8)
    } else {
        corpus().len()
    };
    let mut enc = FieldEncoder::with_feat_dim(t.n_edges(), ling.feature_dim(), cfg.seed);
    enc.eta = 0.12;
    enc.decay = 0.001;
    let tiny = corpus();
    let tiny_hold = holdout_corpus();
    let targets: Vec<Vec<Phasor>> = (0..n_clusters)
        .map(|k| cluster_target_on(&t, k, n_clusters))
        .collect();

    let latest = cfg
        .checkpoint_dir
        .as_ref()
        .map(|d| d.join("encoder-latest.json"));
    let mut steps_done = 0usize;
    let mut report = TrainReport {
        steps: cfg.steps,
        linguistic_source: ling.source().to_string(),
        n_edges: t.n_edges(),
        n_clusters,
        corpus_size: if stream { pool } else { tiny.len() },
        geometry_level: cfg.geometry_level,
        ..TrainReport::default()
    };

    if cfg.resume {
        if let Some(path) = latest.as_ref() {
            if path.exists() {
                let ckpt = load_encoder_checkpoint(path)?;
                if ckpt.version != EncoderCheckpoint::VERSION || ckpt.seed != cfg.seed {
                    return Err(format!(
                        "encoder checkpoint seed/version no coincide ({} vs {})",
                        ckpt.seed, cfg.seed
                    ));
                }
                if ckpt.geometry_level != cfg.geometry_level
                    || ckpt.n_clusters != n_clusters
                    || ckpt.encoder.n_edges != t.n_edges()
                {
                    return Err("checkpoint de otra geometría o número de clusters".into());
                }
                enc = ckpt.encoder;
                steps_done = ckpt.steps_done.min(cfg.steps);
                report = ckpt.report;
                report.steps = cfg.steps;
                report.linguistic_source = ling.source().to_string();
            }
        }
    }

    let pick = |idx: usize| -> Result<(String, usize), String> {
        if stream {
            Ok(example_at(idx as u64, cfg.seed))
        } else {
            let (text, c) = tiny[idx % tiny.len()];
            Ok((text.to_string(), c))
        }
    };

    if steps_done == 0 {
        snapshot_metrics(
            &enc,
            &mut ling,
            &field_cfg,
            &t,
            &targets,
            &tiny,
            &tiny_hold,
            stream,
            pool,
            cfg.seed,
            true,
            &mut report,
        )?;
    }

    let mut rng = Xoshiro256StarStar::seed_from_u64(cfg.seed ^ 0x71);
    for _ in 0..steps_done {
        let _ = rng.gen_range(0..pool.max(1));
    }

    let started = Instant::now();
    let mut last_hour = Instant::now();
    let hour = Duration::from_secs(cfg.checkpoint_every_secs.max(1));
    let limit = if cfg.max_hours > 0.0 {
        Some(Duration::from_secs_f64(cfg.max_hours * 3600.0))
    } else {
        None
    };
    let mut interrupted = false;
    while steps_done < cfg.steps {
        if cfg
            .stop
            .as_ref()
            .is_some_and(|s| s.load(std::sync::atomic::Ordering::SeqCst))
        {
            interrupted = true;
            break;
        }
        if limit.is_some_and(|d| started.elapsed() >= d) {
            break;
        }
        let mut idx = rng.gen_range(0..pool.max(1));
        if stream {
            let mut hops = 0;
            while is_holdout(idx as u64) && hops < 12 {
                idx = rng.gen_range(0..pool.max(1));
                hops += 1;
            }
        }
        let (text, cluster) = pick(idx)?;
        let cluster = cluster % n_clusters;
        let f = ling.features(&text)?;
        let z = enc.encode(&f);
        enc.delta_toward(&f, &z, &targets[cluster]);
        if steps_done % 4 == 3 {
            let z2 = enc.encode(&f);
            let mut psi = FieldState::new(t.clone());
            write_into_field(&mut psi, &z2);
            commit_pattern(&mut psi, &field_cfg);
            relax_free(&mut psi, &field_cfg, cfg.seed + steps_done as u64);
            let saved = enc.eta;
            enc.eta = saved * 0.4;
            let z3 = enc.encode(&f);
            if psi.z.iter().all(|p| p.re.is_finite() && p.im.is_finite()) {
                enc.delta_toward(&f, &z3, &psi.z);
            }
            enc.eta = saved;
        }
        steps_done += 1;
        let hourly = cfg.checkpoint_every_secs > 0 && last_hour.elapsed() >= hour;
        let by_step = cfg.checkpoint_every > 0 && steps_done.is_multiple_of(cfg.checkpoint_every);
        if hourly {
            report.hours_saved = report.hours_saved.saturating_add(1);
            last_hour = Instant::now();
        }
        if cfg.eval_every > 0 && steps_done.is_multiple_of(cfg.eval_every) {
            snapshot_metrics(
                &enc,
                &mut ling,
                &field_cfg,
                &t,
                &targets,
                &tiny,
                &tiny_hold,
                stream,
                pool,
                cfg.seed,
                false,
                &mut report,
            )?;
        }
        if let Some(dir) = cfg.checkpoint_dir.as_ref() {
            if hourly || by_step || steps_done == cfg.steps {
                report.steps_done = steps_done;
                persist_encoder(&cfg, dir, &enc, &report, steps_done)?;
                write_status(dir, &report, started.elapsed(), hourly)?;
            }
        }
        if cfg.log && (steps_done == 1 || steps_done.is_multiple_of(25)) {
            eprintln!(
                "paso {}  aristas={}  clusters={}  acc={:.2}  hold={:.2}  recon={:.3}  fuente={}  {:.0}s",
                steps_done,
                t.n_edges(),
                n_clusters,
                report.decoder_acc_after,
                report.decoder_holdout_after,
                report.recon_after,
                report.linguistic_source,
                started.elapsed().as_secs_f64()
            );
        }
        let _ = field_corpus::topic_name(cluster);
    }

    snapshot_metrics(
        &enc,
        &mut ling,
        &field_cfg,
        &t,
        &targets,
        &tiny,
        &tiny_hold,
        stream,
        pool,
        cfg.seed,
        false,
        &mut report,
    )?;
    report.steps_done = steps_done;
    report.linguistic_source = ling.source().to_string();
    if let Some(dir) = cfg.checkpoint_dir.as_ref() {
        persist_encoder(&cfg, dir, &enc, &report, steps_done)?;
        write_status(dir, &report, started.elapsed(), false)?;
        let summary = serde_json::json!({
            "dataset_id": if stream { "general-enorme" } else { "encoder-corpus" },
            "linguistic_source": report.linguistic_source,
            "interrupted": interrupted,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "topics": N_TOPICS,
            "report": report,
        });
        atomic_write(
            &dir.join("encoder-summary.json"),
            serde_json::to_vec_pretty(&summary)
                .map_err(|e| e.to_string())?
                .as_slice(),
        )?;
    }
    let _ = interrupted;
    Ok((enc, report))
}

fn snapshot_metrics(
    enc: &FieldEncoder,
    ling: &mut LinguisticLayer,
    field_cfg: &FieldConfig,
    t: &ComplexT,
    targets: &[Vec<Phasor>],
    tiny: &[(&str, usize)],
    tiny_hold: &[(&str, usize)],
    stream: bool,
    pool: usize,
    seed: u64,
    before: bool,
    report: &mut TrainReport,
) -> Result<(), String> {
    let (train_items, hold_items) = if stream {
        let tr = sample_indices(pool, 32, seed ^ 0xA1, false);
        let ho = sample_indices(pool, 32, seed ^ 0xB2, true);
        let train: Vec<(String, usize)> = tr.iter().map(|&i| example_at(i, seed)).collect();
        let hold: Vec<(String, usize)> = ho.iter().map(|&i| example_at(i, seed)).collect();
        (train, hold)
    } else {
        (
            tiny.iter().map(|(s, c)| ((*s).to_string(), *c)).collect(),
            tiny_hold
                .iter()
                .map(|(s, c)| ((*s).to_string(), *c))
                .collect(),
        )
    };
    let train_refs: Vec<(&str, usize)> =
        train_items.iter().map(|(s, c)| (s.as_str(), *c)).collect();
    let hold_refs: Vec<(&str, usize)> = hold_items.iter().map(|(s, c)| (s.as_str(), *c)).collect();
    let (zs, labels) = encode_items(enc, ling, &train_refs)?;
    let (same, diff) = cluster_sims(&zs, &labels);
    let (loss, recon) = mean_loss_on(enc, ling, field_cfg, t, &train_refs)?;
    let acc = decoder_accuracy_with(enc, ling, &train_refs, targets)?;
    let hold = decoder_accuracy_with(enc, ling, &hold_refs, targets)?;
    if before {
        report.loss_before = loss;
        report.recon_before = recon;
        report.same_cluster_sim_before = same;
        report.diff_cluster_sim_before = diff;
        report.decoder_acc_before = acc;
        report.decoder_holdout_before = hold;
        report.loss_after = loss;
        report.recon_after = recon;
        report.same_cluster_sim_after = same;
        report.diff_cluster_sim_after = diff;
        report.decoder_acc_after = acc;
        report.decoder_holdout_after = hold;
    } else {
        report.loss_after = loss;
        report.recon_after = recon;
        report.same_cluster_sim_after = same;
        report.diff_cluster_sim_after = diff;
        report.decoder_acc_after = acc;
        report.decoder_holdout_after = hold;
    }
    Ok(())
}

fn mean_loss_on(
    enc: &FieldEncoder,
    ling: &mut LinguisticLayer,
    cfg: &FieldConfig,
    t: &ComplexT,
    data: &[(&str, usize)],
) -> Result<(f64, f64), String> {
    let mask_edges = eval_mask(t.n_edges());
    let mut lt = 0.0;
    let mut rt = 0.0;
    for (i, (text, _)) in data.iter().enumerate() {
        let f = ling.features(text)?;
        let z = enc.encode(&f);
        let mut psi = FieldState::new(t.clone());
        write_into_field(&mut psi, &z);
        commit_pattern(&mut psi, cfg);
        let loss = field_predict_loss(&mut psi, &z, &mask_edges, cfg, 2000 + i as u64);
        lt += loss.total;
        rt += loss.recon;
    }
    let n = data.len().max(1) as f64;
    Ok((lt / n, rt / n))
}

fn persist_encoder(
    cfg: &EncoderTrainConfig,
    dir: &Path,
    enc: &FieldEncoder,
    report: &TrainReport,
    steps_done: usize,
) -> Result<(), String> {
    fs::create_dir_all(dir.join("checkpoints")).map_err(|e| e.to_string())?;
    let ckpt = EncoderCheckpoint {
        version: EncoderCheckpoint::VERSION,
        seed: cfg.seed,
        steps_total: cfg.steps,
        steps_done,
        encoder: enc.clone(),
        report: report.clone(),
        geometry_level: cfg.geometry_level,
        corpus_size: cfg.corpus_size,
        n_clusters: report.n_clusters,
    };
    save_encoder_checkpoint(&dir.join("encoder-latest.json"), &ckpt)?;
    save_encoder_checkpoint(
        &dir.join("checkpoints")
            .join(format!("encoder-step-{steps_done:09}.json")),
        &ckpt,
    )?;
    if report.hours_saved > 0 {
        save_encoder_checkpoint(
            &dir.join("checkpoints")
                .join(format!("hour-{:03}.json", report.hours_saved)),
            &ckpt,
        )?;
    }
    Ok(())
}

fn write_status(
    dir: &Path,
    report: &TrainReport,
    elapsed: Duration,
    hourly: bool,
) -> Result<(), String> {
    let status = serde_json::json!({
        "hours_saved": report.hours_saved,
        "hourly": hourly,
        "steps": report.steps_done,
        "acc": report.decoder_acc_after,
        "hold": report.decoder_holdout_after,
        "recon": report.recon_after,
        "margin": report.margin_after(),
        "n_edges": report.n_edges,
        "n_clusters": report.n_clusters,
        "corpus_size": report.corpus_size,
        "geometry_level": report.geometry_level,
        "source": report.linguistic_source,
        "elapsed_seconds": elapsed.as_secs_f64(),
        "alive": true,
    });
    atomic_write(
        &dir.join("status.json"),
        serde_json::to_vec_pretty(&status)
            .map_err(|e| e.to_string())?
            .as_slice(),
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
        let names = format!("{:?}", crate::field_substrate::FieldState::COMPONENTS);
        assert!(!names.contains("token"));
        let mut psi = FieldState::new(ComplexT::octahedron());
        write_into_field(&mut psi, &z);
        assert_eq!(psi.z.len(), psi.t.n_edges());
        assert_eq!(feature_dim(), GEMMA2_LAYER_COUNT);
    }

    #[test]
    fn same_text_is_deterministic() {
        let mask = LayerSkipMask::from_t21_expensive_skip();
        let a = features_from_text("el perro corre", &mask);
        let b = features_from_text("el perro corre", &mask);
        assert_eq!(a, b);
        assert_eq!(a.len(), GEMMA2_LAYER_COUNT);
    }

    #[test]
    fn cluster_targets_are_geometric_halves() {
        let h = cluster_field_target(0);
        assert_eq!(h.len(), 12);
        let north = h.iter().filter(|p| (p.re - 1.0).abs() < 1e-9).count();
        let south = h.iter().filter(|p| (p.re + 1.0).abs() < 1e-9).count();
        assert_eq!(north, 6);
        assert_eq!(south, 6);
    }

    #[test]
    fn decoder_bank_is_chance_before_training() {
        let enc = FieldEncoder::new(12, 3);
        let mut ling = LinguisticLayer::open(false);
        let acc = decoder_accuracy(&enc, &mut ling, &corpus()).unwrap();
        assert!(
            acc <= 0.5,
            "un mapa aleatorio no debe acertar el banco: {acc}"
        );
    }

    #[test]
    fn hash_probe_does_not_hit_the_bank() {
        let (_enc, report) = train_encoder(120, 0xE4C0);
        assert_eq!(report.linguistic_source, "hash-probe-fallback");
        assert!(
            report.decoder_acc_after < 0.99,
            "hash/n-gram no debe bastar para el banco: {}",
            report.decoder_acc_after
        );
        assert!((report.decoder_holdout_after - report.decoder_holdout_before).abs() < 0.34);
    }

    #[test]
    fn training_separates_clusters_and_hits_the_bank() {
        let dir = std::env::temp_dir().join(format!(
            "enc_field_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (_enc, report) = train_encoder_with(EncoderTrainConfig {
            steps: 120,
            seed: 0xE4C0,
            checkpoint_dir: Some(dir.clone()),
            checkpoint_every: 40,
            resume: false,
            prefer_frozen_gemma: true,
            ..EncoderTrainConfig::default()
        })
        .expect("train");
        if report.linguistic_source != "frozen-gemma2-gguf" {
            return;
        }
        println!(
            "src={} steps={} L {:.4}->{:.4} recon {:.4}->{:.4} same {:.3}->{:.3} diff {:.3}->{:.3} margin {:.3}->{:.3} dec {:.2}->{:.2} hold {:.2}->{:.2}",
            report.linguistic_source,
            report.steps_done,
            report.loss_before,
            report.loss_after,
            report.recon_before,
            report.recon_after,
            report.same_cluster_sim_before,
            report.same_cluster_sim_after,
            report.diff_cluster_sim_before,
            report.diff_cluster_sim_after,
            report.margin_before(),
            report.margin_after(),
            report.decoder_acc_before,
            report.decoder_acc_after,
            report.decoder_holdout_before,
            report.decoder_holdout_after
        );
        assert!(
            report.margin_after() > report.margin_before() + 0.15,
            "encoder should separate clusters in the field: {report:?}"
        );
        assert!(
            (report.decoder_acc_after - 1.0).abs() < 1e-9,
            "decoder should hit the bank: {}",
            report.decoder_acc_after
        );
        assert!(dir.join("encoder-latest.json").exists());
        let ckpt = load_encoder_checkpoint(&dir.join("encoder-latest.json")).unwrap();
        assert_eq!(ckpt.steps_done, 120);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn encoder_checkpoint_resumes() {
        let dir = std::env::temp_dir().join(format!(
            "enc_resume_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let cfg = EncoderTrainConfig {
            steps: 8,
            seed: 0xE4C1,
            checkpoint_dir: Some(dir.clone()),
            checkpoint_every: 4,
            resume: false,
            prefer_frozen_gemma: false,
            ..EncoderTrainConfig::default()
        };
        let (_a, first) = train_encoder_with(cfg.clone()).expect("first");
        assert_eq!(first.steps_done, 8);
        let mut cfg2 = cfg;
        cfg2.steps = 12;
        cfg2.resume = true;
        let (_b, resumed) = train_encoder_with(cfg2).expect("resume");
        assert_eq!(resumed.steps_done, 12);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn frozen_gemma_probe_uses_real_rms_when_gguf_is_present() {
        let mut probe = match FrozenGemmaProbe::try_open() {
            Ok(p) => p,
            Err(_) => return,
        };
        let feats = probe.features("hace frío en la montaña").unwrap();
        assert_eq!(feats.len(), probe.hidden_dim());
        assert!(probe.hidden_dim() > GEMMA2_LAYER_COUNT);
        assert!(feats.iter().all(|v| v.is_finite()));
        let energy: f64 = feats.iter().map(|v| v * v).sum();
        assert!((energy - 1.0).abs() < 1e-6);
        let other = probe.features("el perro corre en el patio").unwrap();
        let mut dot = 0.0;
        for (a, b) in feats.iter().zip(other.iter()) {
            dot += *a * *b;
        }
        assert!(
            dot < 0.98,
            "hidden de frío y perro no deben ser el mismo vector: {dot}"
        );
        let again = probe.features("hace frío en la montaña").unwrap();
        assert_eq!(feats, again);
    }

    #[test]
    fn stream_corpus_trains_on_scaled_substrate() {
        let dir = std::env::temp_dir().join(format!(
            "enc_stream_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (enc, report) = train_encoder_with(EncoderTrainConfig {
            steps: 16,
            seed: 0x51A1,
            checkpoint_dir: Some(dir.clone()),
            checkpoint_every: 8,
            resume: false,
            prefer_frozen_gemma: false,
            geometry_level: 1,
            corpus_size: 200,
            n_clusters: 16,
            eval_every: 16,
            ..EncoderTrainConfig::default()
        })
        .expect("stream");
        assert_eq!(enc.n_edges, 30);
        assert_eq!(report.n_edges, 30);
        assert_eq!(report.n_clusters, 16);
        assert_eq!(report.corpus_size, 200);
        assert!(dir.join("encoder-latest.json").exists());
        assert!(dir.join("status.json").exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
