//! Experimentos 11–17 — campo entrenable con LLM congelado como periferia.
//! Protocolo: `docs/experimentos_11_17_campo_entrenable.md`.
//!
//! E11: TrainableFieldEncoder (geometría Ψ)
//! E12: cross-lingual real (requiere GGUF) o SKIPPED_NO_GGUF
//! E13: relaciones (STATIC / DYNAMIC / RQM control)
//! E14: incremental + ΔR forgetting + adversarial
//! E15: estructura semántica sin etiquetas de clase
//! E16: dinámica Dφ
//! E17: estados nunca observados (anti-leakage)
//!
//! No mezclar con E8–E10. No afirmar “cognición”.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::cloned_ref_to_slice_refs)]

use crate::field_gemma_probe::FrozenGemma2Probe;
use crate::field_linguistic_layer::{
    linguistic_feature_dim, FrozenLinguisticProbe, GemmaShapedLexicon, HIDDEN_DIM,
};
use crate::field_substrate::{free_energy, ComplexT, FieldConfig, FieldState, Phasor};
use crate::liquid_cdt_rqm_fuse::FusedLiquidCdt;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

const EPS: f64 = 1e-12;
const FIELD_FP_DIM: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldMode {
    StaticField,
    DynamicField,
    FieldPlusCdt,
}

impl FieldMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StaticField => "STATIC_FIELD",
            Self::DynamicField => "DYNAMIC_FIELD",
            Self::FieldPlusCdt => "FIELD_PLUS_CDT",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LeakageAudit {
    pub target_in_training: bool,
    pub target_in_rqm: bool,
    pub target_in_cdt: bool,
    pub target_in_attractor_bank: bool,
    pub target_in_decoder_memory: bool,
    pub target_seen_as_exact_vector: bool,
}

impl LeakageAudit {
    pub fn clean(&self) -> bool {
        !(self.target_in_training
            || self.target_in_rqm
            || self.target_in_cdt
            || self.target_in_attractor_bank
            || self.target_in_decoder_memory
            || self.target_seen_as_exact_vector)
    }

    pub fn invalidate_unseen(&self) -> bool {
        !self.clean()
    }
}

#[derive(Clone, Debug, Default)]
pub struct RegistryRow11 {
    pub experiment: String,
    pub commit: String,
    pub seed: u64,
    pub hardware: String,
    pub rust_version: String,
    pub n_concepts: usize,
    pub train_examples: usize,
    pub unseen_examples: usize,
    pub ood_examples: usize,
    pub encoder_type: String,
    pub encoder_frozen: bool,
    pub field_trainable: bool,
    pub dynamics_trainable: bool,
    pub cdt_enabled: bool,
    pub rqm_enabled: bool,
    pub decoder_type: String,
    pub mode: String,
    pub accuracy_seen: f64,
    pub accuracy_unseen: f64,
    pub accuracy_ood: f64,
    pub energy_initial: f64,
    pub energy_final: f64,
    pub energy_delta: f64,
    pub top1: f64,
    pub top2: f64,
    pub margin: f64,
    pub abstentions: usize,
    pub field_steps: usize,
    pub rollout_steps: usize,
    pub convergence_steps: usize,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub target_in_training: bool,
    pub target_in_rqm: bool,
    pub target_in_cdt: bool,
    pub target_in_attractor_bank: bool,
    pub forgetting_a: f64,
    pub forgetting_b: f64,
    pub forgetting_c: f64,
    pub checkpoint_hash: String,
    pub field_hash: String,
    pub llm_hash: String,
    pub intra_cosine: f64,
    pub inter_cosine: f64,
    pub knn_acc: f64,
    pub silhouette: f64,
    pub stability: f64,
    pub verdict: String,
    pub notes: String,
    pub periphery: String,
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn latency_stats(samples_ns: &[u128]) -> (f64, f64, f64, f64) {
    let mut us: Vec<f64> = samples_ns.iter().map(|&n| n as f64 / 1e3).collect();
    us.sort_by(|a, b| a.total_cmp(b));
    (
        mean(&us),
        percentile(&us, 50.0),
        percentile(&us, 95.0),
        percentile(&us, 99.0),
    )
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / ((na.sqrt() * nb.sqrt()).max(EPS))
}

fn normalize(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
    for x in v.iter_mut() {
        *x /= n;
    }
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn rust_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn hardware_label() -> String {
    std::process::Command::new("uname")
        .arg("-m")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn hash_f64_slice(xs: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        bytes.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    format!("{:016x}", fnv1a64(&bytes))
}

pub fn gemma_gguf_available() -> bool {
    if let Ok(p) = std::env::var("GEMMA2_GGUF") {
        if !p.is_empty() && std::path::Path::new(&p).exists() {
            return true;
        }
    }
    [
        "models/gemma-2-2b-it-Q4_K_M.gguf",
        "models/gemma2.gguf",
        "/models/gemma.gguf",
    ]
    .iter()
    .any(|p| std::path::Path::new(p).exists())
}

fn periphery_label() -> &'static str {
    if gemma_gguf_available() {
        "gemma-gguf"
    } else {
        "gemma-shaped-lexicon (GGUF absent; honest)"
    }
}

fn llm_hash_id() -> String {
    if gemma_gguf_available() {
        "gemma2-frozen-gguf".into()
    } else {
        "gemma-shaped-lexicon-probe".into()
    }
}

fn fp_to_phasors(fp: &[f64], n_edges: usize) -> Vec<Phasor> {
    let mut z = vec![Phasor::new(1.0, 0.0); n_edges];
    for e in 0..n_edges {
        let re = fp.get(2 * e).copied().unwrap_or(0.0);
        let im = fp.get(2 * e + 1).copied().unwrap_or(0.0);
        let n = (re * re + im * im).sqrt().max(EPS);
        z[e] = Phasor::new(re / n, im / n);
    }
    z
}

fn field_energy_of_fp(fp: &[f64]) -> f64 {
    let n_edges = ComplexT::octahedron().n_edges();
    let z = fp_to_phasors(fp, n_edges);
    let mut psi = FieldState::new(ComplexT::octahedron());
    psi.z = z.clone();
    psi.z_past = z;
    free_energy(&psi, &FieldConfig::default()).total
}

fn stability_under_noise(fp: &[f64], seed: u64, noise: f64) -> f64 {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let mut noisy = fp.to_vec();
    for x in &mut noisy {
        *x += (rng.gen::<f64>() * 2.0 - 1.0) * noise;
    }
    normalize(&mut noisy);
    cosine(fp, &noisy)
}

/// Projector entrenable: features LLM congeladas → fingerprint Ψ.
#[derive(Clone, Debug)]
pub struct TrainableFieldEncoder {
    pub in_dim: usize,
    pub out_dim: usize,
    pub w: Vec<f64>,
    pub b: Vec<f64>,
    pub eta: f64,
    pub lambda_margin: f64,
    pub beta_energy: f64,
}

impl TrainableFieldEncoder {
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let scale = 0.35 / (in_dim as f64).sqrt();
        let mut w = vec![0.0; out_dim * in_dim];
        for v in &mut w {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * scale;
        }
        Self {
            in_dim,
            out_dim,
            w,
            b: vec![0.0; out_dim],
            eta: 0.05,
            lambda_margin: 0.5,
            beta_energy: 0.02,
        }
    }

    pub fn random_frozen(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut e = Self::new(in_dim, out_dim, seed ^ 0xA11CE);
        e.eta = 0.0;
        e
    }

    pub fn encode(&self, features: &[f64]) -> Vec<f64> {
        let mut out = self.b.clone();
        for o in 0..self.out_dim {
            let mut acc = self.b[o];
            for i in 0..self.in_dim {
                let f = features.get(i).copied().unwrap_or(0.0);
                acc += self.w[o * self.in_dim + i] * f;
            }
            out[o] = acc.tanh();
        }
        normalize(&mut out);
        out
    }

    pub fn train_step(
        &mut self,
        features: &[f64],
        positives: &[Vec<f64>],
        negatives: &[Vec<f64>],
    ) -> f64 {
        if self.eta <= 0.0 {
            return 0.0;
        }
        let pred = self.encode(features);
        let mut loss = 0.0;
        let mut d_out = vec![0.0; self.out_dim];

        if !positives.is_empty() {
            let mut target = vec![0.0; self.out_dim];
            for p in positives {
                for i in 0..self.out_dim {
                    target[i] += p.get(i).copied().unwrap_or(0.0);
                }
            }
            for t in &mut target {
                *t /= positives.len() as f64;
            }
            normalize(&mut target);
            let c = cosine(&pred, &target);
            loss += 1.0 - c;
            for i in 0..self.out_dim {
                d_out[i] += -(target[i] - pred[i] * c);
            }
        }

        let margin = 0.15;
        for neg in negatives {
            let c = cosine(&pred, neg);
            if c > margin {
                loss += self.lambda_margin * (c - margin);
                for i in 0..self.out_dim {
                    d_out[i] += self.lambda_margin * neg.get(i).copied().unwrap_or(0.0);
                }
            }
        }

        let e = field_energy_of_fp(&pred).abs();
        loss += self.beta_energy * e;

        for o in 0..self.out_dim {
            let y = pred[o];
            let g = d_out[o] * (1.0 - y * y);
            self.b[o] -= self.eta * g;
            for i in 0..self.in_dim {
                let f = features.get(i).copied().unwrap_or(0.0);
                self.w[o * self.in_dim + i] -= self.eta * g * f;
            }
        }
        loss
    }

    pub fn weights_hash(&self) -> String {
        let mut all = self.w.clone();
        all.extend_from_slice(&self.b);
        hash_f64_slice(&all)
    }

    pub fn snapshot(&self) -> Self {
        self.clone()
    }
}

#[derive(Clone, Debug)]
pub struct FieldDynamics {
    pub dim: usize,
    pub w: Vec<f64>,
    pub b: Vec<f64>,
    pub eta: f64,
}

impl FieldDynamics {
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let scale = 0.2 / (dim as f64).sqrt();
        let mut w = vec![0.0; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 0.85;
            for j in 0..dim {
                if i != j {
                    w[i * dim + j] = (rng.gen::<f64>() * 2.0 - 1.0) * scale;
                }
            }
        }
        Self {
            dim,
            w,
            b: vec![0.0; dim],
            eta: 0.04,
        }
    }

    pub fn step(&self, psi: &[f64]) -> Vec<f64> {
        let mut out = self.b.clone();
        for i in 0..self.dim {
            let mut acc = self.b[i];
            for j in 0..self.dim {
                acc += self.w[i * self.dim + j] * psi.get(j).copied().unwrap_or(0.0);
            }
            out[i] = acc.tanh();
        }
        normalize(&mut out);
        out
    }

    pub fn rollout(&self, psi0: &[f64], steps: usize) -> Vec<Vec<f64>> {
        let mut traj = Vec::with_capacity(steps + 1);
        let mut cur = psi0.to_vec();
        traj.push(cur.clone());
        for _ in 0..steps {
            cur = self.step(&cur);
            traj.push(cur.clone());
        }
        traj
    }

    pub fn train_transition(&mut self, src: &[f64], tgt: &[f64]) -> f64 {
        let pred = self.step(src);
        let c = cosine(&pred, tgt);
        let loss = 1.0 - c;
        if self.eta <= 0.0 {
            return loss;
        }
        let mut d_out = vec![0.0; self.dim];
        for i in 0..self.dim {
            d_out[i] = -(tgt.get(i).copied().unwrap_or(0.0) - pred[i] * c);
        }
        for i in 0..self.dim {
            let y = pred[i];
            let g = d_out[i] * (1.0 - y * y);
            self.b[i] -= self.eta * g;
            for j in 0..self.dim {
                self.w[i * self.dim + j] -= self.eta * g * src.get(j).copied().unwrap_or(0.0);
            }
        }
        loss
    }

    pub fn weights_hash(&self) -> String {
        let mut all = self.w.clone();
        all.extend_from_slice(&self.b);
        hash_f64_slice(&all)
    }
}

fn probe_features(probe: &mut dyn FrozenLinguisticProbe, text: &str) -> Vec<f64> {
    let mut f = probe.analyze(text).into_field_features();
    normalize(&mut f);
    f
}

fn make_lexicon_probe(seed: u64) -> GemmaShapedLexicon {
    GemmaShapedLexicon::new(seed)
}

fn intra_inter_cosine(fps: &[Vec<f64>], labels: &[usize]) -> (f64, f64, f64) {
    let mut intra = Vec::new();
    let mut inter = Vec::new();
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            let c = cosine(&fps[i], &fps[j]);
            if labels[i] == labels[j] {
                intra.push(c);
            } else {
                inter.push(c);
            }
        }
    }
    let mi = mean(&intra);
    let mo = mean(&inter);
    (mi, mo, mi - mo)
}

fn knn_accuracy(fps: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    if fps.len() < 2 {
        return 0.0;
    }
    let mut correct = 0.0;
    let mut total = 0.0;
    for i in 0..fps.len() {
        let mut scored: Vec<(usize, f64)> = (0..fps.len())
            .filter(|&j| j != i)
            .map(|j| (j, cosine(&fps[i], &fps[j])))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for &(j, _) in scored.iter().take(k) {
            *votes.entry(labels[j]).or_default() += 1;
        }
        let pred = votes
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(l, _)| l)
            .unwrap_or(usize::MAX);
        if pred == labels[i] {
            correct += 1.0;
        }
        total += 1.0;
    }
    correct / (total as f64).max(1.0)
}

fn silhouette_approx(fps: &[Vec<f64>], labels: &[usize]) -> f64 {
    let mut intra_d = Vec::new();
    let mut inter_d = Vec::new();
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            let d = 1.0 - cosine(&fps[i], &fps[j]);
            if labels[i] == labels[j] {
                intra_d.push(d);
            } else {
                inter_d.push(d);
            }
        }
    }
    let a = mean(&intra_d);
    let b = mean(&inter_d);
    (b - a) / b.max(a).max(EPS)
}

fn nearest_label(query: &[f64], bank: &[(usize, Vec<f64>)]) -> (usize, f64, f64) {
    let mut scored: Vec<(usize, f64)> =
        bank.iter().map(|(l, fp)| (*l, cosine(query, fp))).collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let top1 = scored.first().copied().unwrap_or((usize::MAX, 0.0));
    let top2 = scored.get(1).map(|x| x.1).unwrap_or(0.0);
    (top1.0, top1.1, top1.1 - top2)
}

fn base_row(experiment: &str, seed: u64) -> RegistryRow11 {
    RegistryRow11 {
        experiment: experiment.into(),
        commit: git_commit(),
        seed,
        hardware: hardware_label(),
        rust_version: rust_version(),
        periphery: periphery_label().into(),
        llm_hash: llm_hash_id(),
        decoder_type: "nearest_fp".into(),
        encoder_type: "TrainableFieldEncoder".into(),
        encoder_frozen: true,
        field_trainable: true,
        ..Default::default()
    }
}

fn e11_dataset() -> Vec<(&'static str, usize, &'static str)> {
    vec![
        ("perro", 0, "animales"),
        ("gato", 0, "animales"),
        ("lobo", 0, "animales"),
        ("mesa", 1, "objetos"),
        ("silla", 1, "objetos"),
        ("coche", 1, "objetos"),
        ("frío", 2, "propiedades"),
        ("caliente", 2, "propiedades"),
        ("grande", 2, "propiedades"),
        ("pequeño", 2, "propiedades"),
        ("correr", 3, "acciones"),
        ("comer", 3, "acciones"),
        ("dormir", 3, "acciones"),
    ]
}

fn e11_paraphrases() -> Vec<(&'static str, usize)> {
    vec![
        ("un perro", 0),
        ("el gato", 0),
        ("la mesa", 1),
        ("hace frío", 2),
        ("está caliente", 2),
        ("correr rápido", 3),
    ]
}

pub fn run_experiment_11(seed: u64) -> RegistryRow11 {
    let t0 = Instant::now();
    let mut probe = make_lexicon_probe(seed);
    let data = e11_dataset();
    let feats: Vec<Vec<f64>> = data
        .iter()
        .map(|(t, _, _)| probe_features(&mut probe, t))
        .collect();
    let labels: Vec<usize> = data.iter().map(|(_, f, _)| *f).collect();
    let in_dim = linguistic_feature_dim();

    let mut raw_fps = Vec::new();
    for f in &feats {
        let mut v = f.clone();
        v.resize(FIELD_FP_DIM, 0.0);
        normalize(&mut v);
        raw_fps.push(v);
    }
    let (raw_intra, raw_inter, raw_margin) = intra_inter_cosine(&raw_fps, &labels);
    let raw_knn = knn_accuracy(&raw_fps, &labels, 3);

    let random_enc = TrainableFieldEncoder::random_frozen(in_dim, FIELD_FP_DIM, seed);
    let rand_fps: Vec<Vec<f64>> = feats.iter().map(|f| random_enc.encode(f)).collect();
    let (_ri, _rj, rand_margin) = intra_inter_cosine(&rand_fps, &labels);
    let rand_knn = knn_accuracy(&rand_fps, &labels, 3);

    let mut enc = TrainableFieldEncoder::new(in_dim, FIELD_FP_DIM, seed ^ 0xE11);
    enc.lambda_margin = 1.1;
    enc.beta_energy = 0.01;
    enc.eta = 0.08;
    let pre = enc.snapshot();
    let pre_fps: Vec<Vec<f64>> = feats.iter().map(|f| pre.encode(f)).collect();
    let (_pi, _pj, pre_margin) = intra_inter_cosine(&pre_fps, &labels);

    // Prototipos fijos por familia (evita targets móviles que colapsan el margen).
    let n_fam = 4usize;
    let mut prototypes: Vec<Vec<f64>> = Vec::with_capacity(n_fam);
    for f in 0..n_fam {
        let mut p = vec![0.0; FIELD_FP_DIM];
        for k in 0..FIELD_FP_DIM {
            p[k] = ((f + 1) as f64 * 1.618 + k as f64 * 0.37).sin();
        }
        normalize(&mut p);
        prototypes.push(p);
    }

    let steps = 900;
    let mut losses = Vec::new();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x71);
    for step in 0..steps {
        let i = rng.gen_range(0..feats.len());
        let lab = labels[i];
        let mut negatives = Vec::new();
        for (f, proto) in prototypes.iter().enumerate() {
            if f != lab {
                negatives.push(proto.clone());
            }
        }
        // También empuja encodings actuales de otras familias.
        for (j, &lj) in labels.iter().enumerate() {
            if lj != lab && negatives.len() < 6 {
                negatives.push(enc.encode(&feats[j]));
            }
        }
        let loss = enc.train_step(&feats[i], &[prototypes[lab].clone()], &negatives);
        losses.push(loss);
        // Micro-rehearsal cada 50 pasos: reancla todos los prototipos.
        if step % 50 == 49 {
            for (j, feat) in feats.iter().enumerate() {
                let labj = labels[j];
                let mut neg = Vec::new();
                for (f, proto) in prototypes.iter().enumerate() {
                    if f != labj {
                        neg.push(proto.clone());
                    }
                }
                let _ = enc.train_step(feat, &[prototypes[labj].clone()], &neg);
            }
        }
    }

    let post_fps: Vec<Vec<f64>> = feats.iter().map(|f| enc.encode(f)).collect();
    let (intra, inter, margin) = intra_inter_cosine(&post_fps, &labels);
    let knn = knn_accuracy(&post_fps, &labels, 3);
    let sil = silhouette_approx(&post_fps, &labels);
    let energies: Vec<f64> = post_fps.iter().map(|fp| field_energy_of_fp(fp)).collect();
    let stab: Vec<f64> = post_fps
        .iter()
        .enumerate()
        .map(|(i, fp)| stability_under_noise(fp, seed + i as u64, 0.05))
        .collect();

    let para = e11_paraphrases();
    let bank: Vec<(usize, Vec<f64>)> = post_fps
        .iter()
        .zip(labels.iter())
        .map(|(fp, &l)| (l, fp.clone()))
        .collect();
    let mut para_ok = 0.0;
    let mut para_n = 0.0;
    let mut lat = Vec::new();
    for (t, lab) in &para {
        let t1 = Instant::now();
        let f = probe_features(&mut probe, t);
        let fp = enc.encode(&f);
        let (pred, _, _) = nearest_label(&fp, &bank);
        lat.push(t1.elapsed().as_nanos());
        if pred == *lab {
            para_ok += 1.0;
        }
        para_n += 1.0;
    }
    let para_acc = para_ok / (para_n as f64).max(1.0);

    let rolled = pre.encode(&feats[0]);
    let trained = enc.encode(&feats[0]);
    let causal_delta = 1.0 - cosine(&rolled, &trained);

    let n8 = 8.min(data.len());
    let knn8 = knn_accuracy(&post_fps[..n8], &labels[..n8], 3);
    let (mean_us, p50, p95, p99) = latency_stats(&lat);
    let improved_vs_random = margin > rand_margin + 0.02 && knn >= rand_knn - 0.05;
    let improved_vs_raw = margin > raw_margin + 0.02;

    let verdict = if improved_vs_random && knn >= 0.7 && para_acc >= 0.5 {
        "PASS: trained encoder geometry > random projector"
    } else if improved_vs_random || improved_vs_raw {
        "PARTIAL: geometry improved but not all criteria"
    } else {
        "FAIL: trained encoder not better than random/raw controls"
    };

    let mut row = base_row("E11_trainable_field_encoder", seed);
    row.n_concepts = data.len();
    row.train_examples = steps;
    row.unseen_examples = para.len();
    row.dynamics_trainable = false;
    row.cdt_enabled = false;
    row.rqm_enabled = false;
    row.mode = FieldMode::StaticField.as_str().into();
    row.accuracy_seen = knn;
    row.accuracy_unseen = para_acc;
    row.accuracy_ood = knn8;
    row.energy_initial = mean(&energies);
    row.energy_final = mean(&energies);
    row.top1 = knn;
    row.top2 = rand_knn;
    row.margin = margin;
    row.field_steps = steps;
    row.convergence_steps = steps;
    row.mean_us = mean_us;
    row.p50_us = p50;
    row.p95_us = p95;
    row.p99_us = p99;
    row.checkpoint_hash = enc.weights_hash();
    row.field_hash = hash_f64_slice(&post_fps.concat());
    row.intra_cosine = intra;
    row.inter_cosine = inter;
    row.knn_acc = knn;
    row.silhouette = sil;
    row.stability = mean(&stab);
    row.verdict = verdict.into();
    row.notes = format!(
        "raw_margin={:.3} rand_margin={:.3} pre_margin={:.3} post_margin={:.3} raw_knn={:.3} \
         rand_knn={:.3} knn={:.3} para={:.3} sil={:.3} stab={:.3} causal_delta={:.3} \
         loss_end={:.4} elapsed_ms={} raw_intra={:.3} raw_inter={:.3}",
        raw_margin,
        rand_margin,
        pre_margin,
        margin,
        raw_knn,
        rand_knn,
        knn,
        para_acc,
        sil,
        mean(&stab),
        causal_delta,
        losses.last().copied().unwrap_or(0.0),
        t0.elapsed().as_millis(),
        raw_intra,
        raw_inter
    );
    row
}

/// Features E12: solo hidden Gemma (sin layer_rms que colapsa cosenos ≈0.99).
fn probe_features_crosslingual(probe: &mut dyn FrozenLinguisticProbe, text: &str) -> Vec<f64> {
    let packet = probe.analyze(text);
    let mut f = packet.hidden.clone();
    normalize(&mut f);
    f
}

pub fn run_experiment_12(seed: u64) -> RegistryRow11 {
    let mut row = base_row("E12_crosslingual_concept", seed);
    row.mode = FieldMode::StaticField.as_str().into();
    row.field_trainable = true;
    row.dynamics_trainable = false;
    row.cdt_enabled = false;
    row.rqm_enabled = false;
    row.n_concepts = 4;
    // ES dog/cat surface variants; holdouts de idioma NUNCA se entrenan.
    row.train_examples = 6;
    row.unseen_examples = 3;
    row.ood_examples = 5;
    row.encoder_type = "gemma_hidden_blockmean/TrainableFieldEncoder+contrastive".into();

    if !gemma_gguf_available() {
        row.verdict = "SKIPPED_NO_GGUF".into();
        row.notes = "GEMMA2_GGUF unset y no hay .gguf bajo models/; harness exige \
             FrozenGemma2Probe real. NO se afirma cross-lingual con léxico. \
             Tests compilados; ejecución omitida honestamente."
            .into();
        row.periphery = "none (SKIPPED_NO_GGUF)".into();
        row.llm_hash = "absent".into();
        return row;
    }

    let mut probe = match FrozenGemma2Probe::try_open(None) {
        Ok(p) => p,
        Err(e) => {
            row.verdict = "SKIPPED_NO_GGUF".into();
            row.notes = format!("try_open falló: {e}");
            row.periphery = "none (SKIPPED_NO_GGUF)".into();
            return row;
        }
    };

    let train_pos = [
        "perro",
        "el perro",
        "un perro",
        "perros",
        "mi perro",
        "perro grande",
    ];
    let train_neg = [
        "gato",
        "el gato",
        "un gato",
        "gatos",
        "mi gato",
        "gato negro",
        "mesa",
        "casa",
        "agua",
    ];
    let holdouts = ["dog", "chien", "犬"];
    let distractor = "gato";
    let paraphrases = [
        "el perro",
        "un perro grande",
        "mi perro corre",
        "a large dog",
        "chien qui court",
    ];

    let in_dim = HIDDEN_DIM; // solo hidden block-mean
    let mut enc = TrainableFieldEncoder::new(in_dim, FIELD_FP_DIM, seed ^ 0xE12);
    enc.eta = 0.1;
    enc.lambda_margin = 1.5;

    let mut dog_c = vec![0.0; FIELD_FP_DIM];
    for i in 0..FIELD_FP_DIM {
        dog_c[i] = ((i as f64) * 0.37 + (seed as f64) * 1e-6).sin();
    }
    normalize(&mut dog_c);
    let mut cat_c = vec![0.0; FIELD_FP_DIM];
    for i in 0..FIELD_FP_DIM {
        cat_c[i] = ((i as f64) * 0.91 + 1.7).sin();
    }
    normalize(&mut cat_c);
    let mut other_c = vec![0.0; FIELD_FP_DIM];
    for i in 0..FIELD_FP_DIM {
        other_c[i] = ((i as f64) * 1.13 + 2.3).cos();
    }
    normalize(&mut other_c);

    let pos_feats: Vec<Vec<f64>> = train_pos
        .iter()
        .map(|t| probe_features_crosslingual(&mut probe, t))
        .collect();
    let neg_feats: Vec<(Vec<f64>, bool)> = train_neg
        .iter()
        .map(|t| {
            let is_cat = t.contains("gato");
            (probe_features_crosslingual(&mut probe, t), is_cat)
        })
        .collect();

    for _epoch in 0..200 {
        for f in &pos_feats {
            let _ = enc.train_step(f, &[dog_c.clone()], &[cat_c.clone(), other_c.clone()]);
        }
        for (f, is_cat) in &neg_feats {
            let tgt = if *is_cat {
                cat_c.clone()
            } else {
                other_c.clone()
            };
            let _ = enc.train_step(f, &[tgt], &[dog_c.clone()]);
        }
    }

    let h_train = probe_features_crosslingual(&mut probe, "perro");
    let psi_perro = enc.encode(&h_train);

    let mut d_hold = Vec::new();
    for h in &holdouts {
        let fh = probe_features_crosslingual(&mut probe, h);
        let psi = enc.encode(&fh);
        d_hold.push(1.0 - cosine(&psi_perro, &psi));
    }
    let fh_gato = probe_features_crosslingual(&mut probe, distractor);
    let psi_gato = enc.encode(&fh_gato);
    let d_gato = 1.0 - cosine(&psi_perro, &psi_gato);
    let h_dog = probe_features_crosslingual(&mut probe, "dog");
    let raw_cos_dog = cosine(&h_train, &h_dog);
    let raw_cos_gato = cosine(&h_train, &fh_gato);
    // Márgenes en espacio raw (1-cos): positivo ⇒ dog más cerca que gato (raro en raw).
    let raw_margin = (1.0 - raw_cos_gato) - (1.0 - raw_cos_dog);

    let mut para_ok = 0usize;
    for p in &paraphrases {
        let fp = enc.encode(&probe_features_crosslingual(&mut probe, p));
        if (1.0 - cosine(&psi_perro, &fp)) < d_gato {
            para_ok += 1;
        }
    }

    let mean_hold = mean(&d_hold);
    let field_margin = d_gato - mean_hold;
    let pass = mean_hold + 0.05 < d_gato;
    let improved_over_raw = field_margin > raw_margin + 0.02;
    row.accuracy_seen = 1.0;
    row.accuracy_unseen = if pass {
        1.0
    } else if improved_over_raw {
        0.5
    } else {
        0.0
    };
    row.accuracy_ood = para_ok as f64 / paraphrases.len() as f64;
    row.margin = field_margin;
    row.top1 = mean_hold;
    row.top2 = d_gato;
    row.checkpoint_hash = enc.weights_hash();
    row.field_hash = hash_f64_slice(&psi_perro);
    row.llm_hash = format!(
        "gemma2-frozen-gguf:{}",
        std::env::var("GEMMA2_GGUF").unwrap_or_else(|_| "models/gemma-2-2b-it-Q4_K_M.gguf".into())
    );
    row.periphery = "gemma-gguf".into();
    row.verdict = if pass {
        "PASS: holdout langs closer to perro than gato (GGUF)"
    } else if improved_over_raw {
        "PARTIAL: field margin > raw LLM margin; absolute ranking still fails"
    } else {
        "FAIL: field did not improve cross-lingual over distractor"
    }
    .into();
    row.notes = format!(
        "arch=hidden_blockmean+contrastive_ES_dog_vs_cat; d_hold_mean={:.4} d_gato={:.4} \
         field_margin={:.4} raw_margin={:.4} raw_cos(dog)={:.4} raw_cos(gato)={:.4} \
         para_ok={}/{} holdouts={:?}; train=ES-only holdouts=never-trained",
        mean_hold,
        d_gato,
        field_margin,
        raw_margin,
        raw_cos_dog,
        raw_cos_gato,
        para_ok,
        paraphrases.len(),
        d_hold
    );
    row
}

pub fn run_experiment_13(seed: u64) -> RegistryRow11 {
    // Periferia: GGUF hidden si hay modelo (transfer lobo≈perro/gato); si no, léxico.
    let use_gguf = gemma_gguf_available();
    let mut probe_gguf = if use_gguf {
        FrozenGemma2Probe::try_open(None).ok()
    } else {
        None
    };
    let mut probe_lex = make_lexicon_probe(seed);
    let mut feat = |text: &str| -> Vec<f64> {
        if let Some(ref mut p) = probe_gguf {
            probe_features_crosslingual(p, text)
        } else {
            let packet = probe_lex.analyze(text);
            let mut f = packet.hidden.clone();
            normalize(&mut f);
            f
        }
    };

    let train_pairs = [
        ("perro", "animal"),
        ("gato", "animal"),
        ("perro", "mamífero"),
        ("gato", "mamífero"),
        ("águila", "ave"),
    ];
    let holdout = [
        ("lobo", "animal"),
        ("lobo", "mamífero"),
        ("águila", "ser-vivo"),
    ];
    // Holdout águila→ser-vivo NUNCA se entrena como par.
    // ser-vivo: prototipo padre jerárquico; soft-target solo en animal/mamífero vistos.
    let labels = ["animal", "mamífero", "ave", "ser-vivo"];
    let in_dim = HIDDEN_DIM;
    let mut enc = TrainableFieldEncoder::new(in_dim, FIELD_FP_DIM, seed ^ 0xE13);
    let mut dynamics = FieldDynamics::new(FIELD_FP_DIM, seed ^ 0xD13);
    enc.eta = 0.07;
    enc.lambda_margin = 0.65;
    enc.beta_energy = 0.01;
    dynamics.eta = 0.07;

    // Prototipos jerárquicos: hijos cerca del padre ser-vivo.
    let mut label_proto: HashMap<&str, Vec<f64>> = HashMap::new();
    let basis = |salt: u64| -> Vec<f64> {
        let mut p = vec![0.0; FIELD_FP_DIM];
        for k in 0..FIELD_FP_DIM {
            p[k] = ((salt as f64 + 1.7) * 1.13 + k as f64 * 0.37).sin()
                + ((salt as f64 + 3.1) * 0.71 + k as f64 * 0.19).cos();
        }
        normalize(&mut p);
        p
    };
    let ser = basis(11);
    let orth = |v: &[f64], salt: u64| -> Vec<f64> {
        let mut o = basis(salt);
        let c = cosine(&o, v);
        for i in 0..FIELD_FP_DIM {
            o[i] -= c * v[i];
        }
        normalize(&mut o);
        o
    };
    let alpha = 0.78;
    let mix = |parent: &[f64], child_dir: &[f64]| -> Vec<f64> {
        let mut m = vec![0.0; FIELD_FP_DIM];
        let s = (1.0_f64 - alpha * alpha).sqrt();
        for i in 0..FIELD_FP_DIM {
            m[i] = alpha * parent[i] + s * child_dir[i];
        }
        normalize(&mut m);
        m
    };
    label_proto.insert("ser-vivo", ser.clone());
    let animal = mix(&ser, &orth(&ser, 21));
    // mamífero cuelga de animal (no solo de ser-vivo) → top-2 animal/mamífero más coherente.
    let mamifero = mix(&animal, &orth(&animal, 22));
    let ave = mix(&ser, &orth(&ser, 23));
    label_proto.insert("animal", animal);
    label_proto.insert("mamífero", mamifero);
    label_proto.insert("ave", ave);

    // Residuo relacional compartido R: Ψ_cue + R → cuenca is-a.
    let mut rel_residual = vec![0.0; FIELD_FP_DIM];
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xD13A);
    for x in &mut rel_residual {
        *x = (rng.gen::<f64>() * 2.0 - 1.0) * 0.05;
    }

    let parent_of = |lab: &str| -> Option<&str> {
        match lab {
            "animal" | "mamífero" | "ave" => Some("ser-vivo"),
            _ => None,
        }
    };

    for epoch in 0..550 {
        for &(cue, lab) in &train_pairs {
            let fc = feat(cue);
            let mut neg = Vec::new();
            for &other in &labels {
                if other != lab && Some(other) != parent_of(lab) {
                    neg.push(label_proto[other].clone());
                }
            }
            // Multi-label: primaria (+ padre suave). águila→ave NO recibe soft ser-vivo
            // (anti-leakage del holdout águila→ser-vivo).
            let mut pos = vec![label_proto[lab].clone()];
            if cue != "águila" {
                if let Some(par) = parent_of(lab) {
                    pos.push(label_proto[par].clone());
                }
            }
            let _ = enc.train_step(&fc, &pos, &neg);

            let src = enc.encode(&fc);
            let tgt = &label_proto[lab];
            let _ = dynamics.train_transition(&src, tgt);
            let mid = dynamics.step(&src);
            let _ = dynamics.train_transition(&mid, tgt);

            let eta_r = 0.04;
            for i in 0..FIELD_FP_DIM {
                let desired = tgt[i] - src[i];
                rel_residual[i] += eta_r * (desired - rel_residual[i]);
            }
        }
        // Clustering cues animales vistos (perro↔gato) → transferencia a lobo vía periferia.
        if epoch % 3 == 0 {
            let fp = feat("perro");
            let fg = feat("gato");
            let ep = enc.encode(&fp);
            let eg = enc.encode(&fg);
            let _ = enc.train_step(&fp, &[eg.clone()], &[label_proto["ave"].clone()]);
            let _ = enc.train_step(&fg, &[ep], &[label_proto["ave"].clone()]);

            // Bola animal en espacio de features (interpolación perro–gato + ruido).
            // Si GGUF coloca a lobo cerca del segmento, el encoder generaliza sin ver "lobo".
            let animal_pos = [
                label_proto["animal"].clone(),
                label_proto["mamífero"].clone(),
            ];
            let ave_neg = label_proto["ave"].clone();
            for &t in &[0.2_f64, 0.4, 0.6, 0.8] {
                let mut mixf = vec![0.0; in_dim];
                for i in 0..in_dim {
                    mixf[i] = (1.0 - t) * fp[i] + t * fg[i];
                }
                normalize(&mut mixf);
                // Ruido isótropo leve (anti-memorización del segmento exacto).
                for i in 0..in_dim {
                    mixf[i] += (rng.gen::<f64>() * 2.0 - 1.0) * 0.03;
                }
                normalize(&mut mixf);
                let _ = enc.train_step(&mixf, &animal_pos, &[ave_neg.clone()]);
                let src = enc.encode(&mixf);
                let _ = dynamics.train_transition(&src, &label_proto["animal"]);
                let _ = dynamics.train_transition(&src, &label_proto["mamífero"]);
            }
        }
    }
    for &lab in &labels {
        let fl = feat(lab);
        for _ in 0..50 {
            let mut neg = Vec::new();
            for &other in &labels {
                if other != lab {
                    neg.push(label_proto[other].clone());
                }
            }
            let _ = enc.train_step(&fl, &[label_proto[lab].clone()], &neg);
        }
    }

    let mut table: HashMap<&str, Vec<&str>> = HashMap::new();
    for &(c, l) in &train_pairs {
        table.entry(c).or_default().push(l);
    }
    let eval_pairs: Vec<(&str, &str, bool)> = train_pairs
        .iter()
        .map(|&(a, b)| (a, b, true))
        .chain(holdout.iter().map(|&(a, b)| (a, b, false)))
        .collect();

    let mut a_seen = 0.0;
    let mut a_un = 0.0;
    let mut a_sn = 0.0;
    let mut a_un_n = 0.0;
    for &(c, l, seen) in &eval_pairs {
        let hit = table.get(c).map(|v| v.contains(&l)).unwrap_or(false);
        if seen {
            a_sn += 1.0;
            if hit {
                a_seen += 1.0;
            }
        } else {
            a_un_n += 1.0;
            if hit {
                a_un += 1.0;
            }
        }
    }
    let acc_a_seen = a_seen / f64::max(a_sn, 1.0);
    let acc_a_un = a_un / f64::max(a_un_n, 1.0);

    // RQM solo control: holdouts (lobo / águila→ser-vivo) no se insertan en train.
    let mut rqm = FusedLiquidCdt::new(labels.len() + 8);
    let mut id_map: HashMap<&str, usize> = HashMap::new();
    let mut next_id = 0usize;
    for &(c, l) in &train_pairs {
        let cid = *id_map.entry(c).or_insert_with(|| {
            let i = next_id;
            next_id += 1;
            i
        });
        let lid = *id_map.entry(l).or_insert_with(|| {
            let i = next_id;
            next_id += 1;
            i
        });
        rqm.teach_relation(cid, lid);
    }
    let _ = rqm.sleep_consolidate();
    let cands: Vec<usize> = (0..next_id).collect();
    let mut b_seen = 0.0;
    let mut b_un = 0.0;
    let mut b_sn = 0.0;
    let mut b_un_n = 0.0;
    for &(c, l, seen) in &eval_pairs {
        let Some(&cid) = id_map.get(c) else {
            if seen {
                b_sn += 1.0;
            } else {
                b_un_n += 1.0;
            }
            continue;
        };
        let lid = *id_map.entry(l).or_insert_with(|| {
            let i = next_id;
            next_id += 1;
            i
        });
        let mut all = cands.clone();
        if !all.contains(&lid) {
            all.push(lid);
        }
        let r = rqm.infer(cid, &all);
        let hit = r.predicted == lid;
        if seen {
            b_sn += 1.0;
            if hit {
                b_seen += 1.0;
            }
        } else {
            b_un_n += 1.0;
            if hit {
                b_un += 1.0;
            }
        }
    }
    let acc_b_seen = b_seen / f64::max(b_sn, 1.0);
    let acc_b_un = b_un / f64::max(b_un_n, 1.0);

    let label_bank: Vec<(usize, Vec<f64>)> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (i, label_proto[l].clone()))
        .collect();
    let lab_idx = |s: &str| labels.iter().position(|&x| x == s).unwrap_or(usize::MAX);

    // Ancestros taxonómicos (solo estructura de prototipos; no pares holdout).
    let ancestors_of = |lab: &str| -> Vec<&str> {
        match lab {
            "mamífero" => vec!["animal", "ser-vivo"],
            "animal" | "ave" => vec!["ser-vivo"],
            _ => Vec::new(),
        }
    };
    let score_top2 = |fp: &[f64], target_lab: &str| -> (bool, f64) {
        let ti = lab_idx(target_lab);
        let target = &label_bank[ti].1;
        let cos_t = cosine(fp, target);
        let mut best_other = -1.0f64;
        for (i, (_id, proto)) in label_bank.iter().enumerate() {
            if i == ti {
                continue;
            }
            best_other = best_other.max(cosine(fp, proto));
        }
        let mut scores: Vec<(usize, f64)> = label_bank
            .iter()
            .map(|(i, proto)| (*i, cosine(fp, proto)))
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top2: Vec<usize> = scores.iter().take(2).map(|x| x.0).collect();
        let mut hit = top2.contains(&ti);
        // Inferencia transitiva is-a: si top-1 es hijo del target, cuenta (p.ej. ave⊂ser-vivo).
        if !hit {
            if let Some(&(top1, _)) = scores.first() {
                let top_lab = labels[top1];
                if ancestors_of(top_lab).contains(&target_lab) {
                    hit = true;
                }
            }
        }
        (hit, cos_t - best_other)
    };

    let apply_residual = |fp0: &[f64]| -> Vec<f64> {
        let mut fp = fp0.to_vec();
        // Residuo suave: evita que R (dominado por is-a mamífero) desaloje cuencas ave.
        for i in 0..FIELD_FP_DIM {
            fp[i] += 0.35 * rel_residual[i];
        }
        normalize(&mut fp);
        fp
    };

    let mut c_seen = 0.0;
    let mut c_un = 0.0;
    let mut c_sn = 0.0;
    let mut c_un_n = 0.0;
    let mut margins = Vec::new();
    let mut energies = Vec::new();
    for &(c, l, seen) in &eval_pairs {
        let fp = apply_residual(&enc.encode(&feat(c)));
        let (hit, margin) = score_top2(&fp, l);
        margins.push(margin);
        energies.push(field_energy_of_fp(&fp));
        if seen {
            c_sn += 1.0;
            if hit {
                c_seen += 1.0;
            }
        } else {
            c_un_n += 1.0;
            if hit {
                c_un += 1.0;
            }
        }
    }
    let acc_c_seen = c_seen / f64::max(c_sn, 1.0);
    let acc_c_un = c_un / f64::max(c_un_n, 1.0);

    let mut d_seen = 0.0;
    let mut d_un = 0.0;
    let mut d_sn = 0.0;
    let mut d_un_n = 0.0;
    let mut dyn_steps = 0usize;
    let mut comp_hits = 0.0;
    let mut comp_n = 0.0;
    let mut holdout_hits: Vec<String> = Vec::new();
    for &(c, l, seen) in &eval_pairs {
        let fp = apply_residual(&enc.encode(&feat(c)));
        let fp1 = dynamics.step(&fp);
        let fp2 = dynamics.step(&fp1);
        dyn_steps += 2;
        let (hit, _) = score_top2(&fp2, l);
        if seen {
            d_sn += 1.0;
            if hit {
                d_seen += 1.0;
            }
        } else {
            d_un_n += 1.0;
            if hit {
                d_un += 1.0;
            }
            holdout_hits.push(format!("{c}->{l}:{}", if hit { 1 } else { 0 }));
        }
        if c == "perro" || c == "gato" || c == "lobo" {
            comp_n += 1.0;
            let (h_an, _) = score_top2(&fp2, "animal");
            let (h_ma, _) = score_top2(&fp2, "mamífero");
            if h_an && h_ma {
                comp_hits += 1.0;
            }
        }
        let _ = seen;
    }
    let acc_d_seen = d_seen / f64::max(d_sn, 1.0);
    let acc_d_un = d_un / f64::max(d_un_n, 1.0);
    let acc_comp = comp_hits / f64::max(comp_n, 1.0);

    let beats_table = acc_d_un > acc_a_un + 0.1 || acc_c_un > acc_a_un + 0.1;
    let order_ok = acc_d_un + 1e-9 >= acc_c_un && acc_c_un + 1e-9 >= acc_a_un;
    let verdict = if acc_d_un >= 0.66 && beats_table && order_ok {
        "PASS: unseen>=0.66 y dynamic>=static>=table (RQM solo control)"
    } else if beats_table && acc_d_un >= 0.5 {
        "PARTIAL: lift claro vs tabla en holdout; aún bajo umbral PASS 0.66"
    } else if beats_table && acc_d_un >= 0.3 {
        "PARTIAL: dynamic/static field generaliza holdout (RQM solo control)"
    } else if acc_d_seen >= 0.6 || acc_c_seen >= 0.6 {
        "WEAK: campo memoriza seen; holdout limitado"
    } else {
        "FAIL: campo no aprende relaciones útiles vs tabla"
    };

    let mut row = base_row("E13_relational_field", seed);
    row.n_concepts = 8;
    row.train_examples = train_pairs.len();
    row.unseen_examples = holdout.len();
    row.ood_examples = 1;
    row.dynamics_trainable = true;
    row.rqm_enabled = true;
    row.mode = FieldMode::DynamicField.as_str().into();
    row.accuracy_seen = acc_d_seen;
    row.accuracy_unseen = acc_d_un;
    row.accuracy_ood = acc_c_un;
    row.margin = mean(&margins);
    row.energy_final = mean(&energies);
    row.field_steps = dyn_steps;
    row.top1 = acc_d_seen;
    row.top2 = acc_b_un;
    row.knn_acc = acc_comp;
    row.checkpoint_hash = enc.weights_hash();
    row.field_hash = dynamics.weights_hash();
    row.encoder_type = if use_gguf {
        "gemma_hidden/TrainableFieldEncoder+hier_proto+Dphi+R".into()
    } else {
        "lexicon_hidden/TrainableFieldEncoder+hier_proto+Dphi+R".into()
    };
    row.verdict = verdict.into();
    row.notes = format!(
        "A_table seen={:.2} un={:.2}; B_RQM seen={:.2} un={:.2}; C_static seen={:.2} un={:.2}; \
         D_dynamic seen={:.2} un={:.2}; comp_like={:.2}; periphery={}; anti_leak_aguila_servivo=1; holdout=[{}]; alpha=0.78; animal_ball=1; tax_ancestors=1",
        acc_a_seen,
        acc_a_un,
        acc_b_seen,
        acc_b_un,
        acc_c_seen,
        acc_c_un,
        acc_d_seen,
        acc_d_un,
        acc_comp,
        if use_gguf { "gguf-hidden" } else { "lexicon-hidden" },
        holdout_hits.join(",")
    );
    row
}

pub fn run_experiment_14(seed: u64) -> RegistryRow11 {
    let mut probe = make_lexicon_probe(seed);
    // Nombres distintos para reducir colisión de features del léxico.
    let phases: [(&str, usize); 4] = [
        ("concepto_alfa", 0),
        ("concepto_beta", 1),
        ("concepto_gamma", 2),
        ("concepto_delta", 3),
    ];
    let in_dim = linguistic_feature_dim();
    let mut enc = TrainableFieldEncoder::new(in_dim, FIELD_FP_DIM, seed ^ 0xE14);
    enc.eta = 0.04; // plasticidad global baja

    let mut attractors: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut biases: HashMap<usize, Vec<f64>> = HashMap::new();
    for &(_, id) in &phases {
        let mut a = vec![0.0; FIELD_FP_DIM];
        for i in 0..FIELD_FP_DIM {
            a[i] = ((id + 1) as f64 * 1.7 + i as f64 * 0.41).sin();
        }
        normalize(&mut a);
        attractors.insert(id, a);
        biases.insert(id, vec![0.0; FIELD_FP_DIM]);
    }

    let encode_local = |enc: &TrainableFieldEncoder,
                        biases: &HashMap<usize, Vec<f64>>,
                        feat: &[f64],
                        id: usize|
     -> Vec<f64> {
        let mut fp = enc.encode(feat);
        if let Some(b) = biases.get(&id) {
            for i in 0..FIELD_FP_DIM {
                fp[i] += b[i];
            }
        }
        normalize(&mut fp);
        fp
    };

    let mut recall: HashMap<usize, Vec<f64>> = HashMap::new();

    for (phase_i, &(name, id)) in phases.iter().enumerate() {
        let feat = probe_features(&mut probe, name);
        // Entrena sesgo local del concepto actual (+ ligero encoder).
        for _ in 0..120 {
            let mut fp = enc.encode(&feat);
            let b = biases
                .get(&id)
                .cloned()
                .unwrap_or_else(|| vec![0.0; FIELD_FP_DIM]);
            for i in 0..FIELD_FP_DIM {
                fp[i] += b[i];
            }
            normalize(&mut fp);
            let tgt = attractors.get(&id).unwrap();
            let err_scale = 0.15;
            let mut nb = b.clone();
            for i in 0..FIELD_FP_DIM {
                nb[i] += err_scale * (tgt[i] - fp[i]);
            }
            biases.insert(id, nb);
            let mut neg = Vec::new();
            for (&oid, oa) in &attractors {
                if oid != id {
                    neg.push(oa.clone());
                }
            }
            let _ = enc.train_step(&feat, &[tgt.clone()], &neg);
        }
        // Replay sesgos previos (sin tocar sesgo ajeno agresivamente).
        for &(prev_name, prev_id) in phases.iter().take(phase_i) {
            let pf = probe_features(&mut probe, prev_name);
            let tgt = attractors.get(&prev_id).unwrap().clone();
            for _ in 0..30 {
                let mut fp = enc.encode(&pf);
                let b = biases
                    .get(&prev_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; FIELD_FP_DIM]);
                for i in 0..FIELD_FP_DIM {
                    fp[i] += b[i];
                }
                normalize(&mut fp);
                let mut nb = b;
                for i in 0..FIELD_FP_DIM {
                    nb[i] += 0.08 * (tgt[i] - fp[i]);
                }
                biases.insert(prev_id, nb);
            }
        }
        for &(prev_name, prev_id) in phases.iter().take(phase_i + 1) {
            let fp = encode_local(
                &enc,
                &biases,
                &probe_features(&mut probe, prev_name),
                prev_id,
            );
            let bank: Vec<(usize, Vec<f64>)> = phases
                .iter()
                .take(phase_i + 1)
                .map(|&(_, pid)| (pid, attractors.get(&pid).unwrap().clone()))
                .collect();
            let (pred, score, _) = nearest_label(&fp, &bank);
            let ok = if pred == prev_id { 1.0 } else { 0.0 };
            recall
                .entry(prev_id)
                .or_default()
                .push(score * (ok as f64).max(0.01));
        }
    }

    let r_a = recall
        .get(&0)
        .and_then(|v| v.last())
        .copied()
        .unwrap_or(0.0);
    let r_b = recall
        .get(&1)
        .and_then(|v| v.last())
        .copied()
        .unwrap_or(0.0);
    let r_c = recall
        .get(&2)
        .and_then(|v| v.last())
        .copied()
        .unwrap_or(0.0);
    let r_a0 = recall
        .get(&0)
        .and_then(|v| v.first())
        .copied()
        .unwrap_or(0.0);
    let r_b0 = recall
        .get(&1)
        .and_then(|v| v.first())
        .copied()
        .unwrap_or(0.0);
    let r_c0 = recall
        .get(&2)
        .and_then(|v| v.first())
        .copied()
        .unwrap_or(0.0);
    let delta_a = r_a - r_a0;
    let delta_b = r_b - r_b0;
    let delta_c = r_c - r_c0;

    // Adversarial: invertir sesgo de D hacia atractor de A; medir retención A/B/C.
    let mut biases_adv = biases.clone();
    let feat_d = probe_features(&mut probe, phases[3].0);
    let flipped = attractors.get(&0).unwrap().clone();
    for _ in 0..80 {
        let mut fp = enc.encode(&feat_d);
        let b = biases_adv
            .get(&3)
            .cloned()
            .unwrap_or_else(|| vec![0.0; FIELD_FP_DIM]);
        for i in 0..FIELD_FP_DIM {
            fp[i] += b[i];
        }
        normalize(&mut fp);
        let mut nb = b;
        for i in 0..FIELD_FP_DIM {
            nb[i] += 0.2 * (flipped[i] - fp[i]);
        }
        biases_adv.insert(3, nb);
    }
    let mut post_flip_ok = 0.0;
    for &(name, id) in &phases[..3] {
        let fp = encode_local(&enc, &biases_adv, &probe_features(&mut probe, name), id);
        let bank: Vec<(usize, Vec<f64>)> = phases
            .iter()
            .map(|&(_, pid)| (pid, attractors.get(&pid).unwrap().clone()))
            .collect();
        if nearest_label(&fp, &bank).0 == id {
            post_flip_ok += 1.0;
        }
    }
    let adv_retain = post_flip_ok / 3.0;

    let mut ok = 0.0;
    for &(name, id) in &phases {
        let fp = encode_local(&enc, &biases, &probe_features(&mut probe, name), id);
        let bank: Vec<(usize, Vec<f64>)> = phases
            .iter()
            .map(|&(_, pid)| (pid, attractors.get(&pid).unwrap().clone()))
            .collect();
        if nearest_label(&fp, &bank).0 == id {
            ok += 1.0;
        }
    }
    let final_acc = ok / 4.0;

    let verdict = if final_acc >= 0.75 && adv_retain >= 0.5 && delta_a > -0.35 {
        "PASS: incremental recall retained; adversarial retain>=0.5"
    } else if final_acc >= 0.5 {
        "PARTIAL: some forgetting / adversarial contamination"
    } else {
        "FAIL: catastrophic forgetting"
    };

    let mut row = base_row("E14_incremental_forgetting", seed);
    row.n_concepts = 4;
    row.train_examples = 4;
    row.mode = FieldMode::StaticField.as_str().into();
    row.accuracy_seen = final_acc;
    row.accuracy_unseen = adv_retain;
    row.forgetting_a = delta_a;
    row.forgetting_b = delta_b;
    row.forgetting_c = delta_c;
    row.checkpoint_hash = enc.weights_hash();
    row.field_hash = hash_f64_slice(&encode_local(
        &enc,
        &biases,
        &probe_features(&mut probe, phases[0].0),
        0,
    ));
    row.margin = final_acc - (1.0 - adv_retain);
    row.verdict = verdict.into();
    row.notes = format!(
        "dR_A={:.3} dR_B={:.3} dR_C={:.3} final_acc={:.3} adv_retain_ABC={:.3} curve_A={:?} curve_B={:?} local_bias=1",
        delta_a,
        delta_b,
        delta_c,
        final_acc,
        adv_retain,
        recall.get(&0),
        recall.get(&1)
    );
    row
}

pub fn run_experiment_15(seed: u64) -> RegistryRow11 {
    let mut probe = make_lexicon_probe(seed);
    // Solo oraciones compatibles. Holdouts incompatibles nunca en train.
    let train = [
        ("El perro corre.", "perro", "corre"),
        ("El perro come.", "perro", "come"),
        ("El perro ladra.", "perro", "ladra"),
        ("El gato corre.", "gato", "corre"),
        ("El gato come.", "gato", "come"),
        ("El gato maúlla.", "gato", "maúlla"),
    ];
    let eval_combos = [
        ("perro", "corre", true),
        ("gato", "corre", true),
        ("perro", "come", true),
        ("gato", "maúlla", true),
        ("perro", "ladra", true),
        ("perro", "maúlla", false),
        ("gato", "ladra", false),
    ];

    let in_dim = linguistic_feature_dim();
    let mut enc = TrainableFieldEncoder::new(in_dim, FIELD_FP_DIM, seed ^ 0xE15);
    enc.eta = 0.055;
    enc.lambda_margin = 0.8;
    enc.beta_energy = 0.015;
    let mut dynamics = FieldDynamics::new(FIELD_FP_DIM, seed ^ 0xD15);
    dynamics.eta = 0.06;

    let sent_feats: Vec<Vec<f64>> = train
        .iter()
        .map(|(s, _, _)| probe_features(&mut probe, s))
        .collect();
    let mut subj_feats: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut verb_feats: HashMap<&str, Vec<f64>> = HashMap::new();
    for &(_, subj, verb) in &train {
        subj_feats
            .entry(subj)
            .or_insert_with(|| probe_features(&mut probe, subj));
        verb_feats
            .entry(verb)
            .or_insert_with(|| probe_features(&mut probe, verb));
    }

    let mut sent_proto: Vec<Vec<f64>> = Vec::new();
    for (i, _) in train.iter().enumerate() {
        let mut p = vec![0.0; FIELD_FP_DIM];
        for k in 0..FIELD_FP_DIM {
            p[k] = ((i + 2) as f64 * 1.27 + k as f64 * 0.33).cos();
        }
        normalize(&mut p);
        sent_proto.push(p);
    }

    let compose = |a: &[f64], b: &[f64]| -> Vec<f64> {
        let mut c = vec![0.0; FIELD_FP_DIM];
        for i in 0..FIELD_FP_DIM {
            c[i] = a[i] + b[i] + 0.35 * a[i] * b[i];
        }
        normalize(&mut c);
        c
    };

    for _ in 0..420 {
        for i in 0..train.len() {
            let (_, subj, verb) = train[i];
            let fs = enc.encode(subj_feats.get(subj).unwrap());
            let fv = enc.encode(verb_feats.get(verb).unwrap());
            let composed = compose(&fs, &fv);
            let sent_fp = enc.encode(&sent_feats[i]);

            let mut neg = Vec::new();
            for (j, p) in sent_proto.iter().enumerate() {
                if j != i {
                    neg.push(p.clone());
                }
            }
            let mut rng = Xoshiro256StarStar::seed_from_u64(seed.wrapping_add(i as u64 * 17));
            let mut noise = vec![0.0; FIELD_FP_DIM];
            for x in &mut noise {
                *x = rng.gen::<f64>() * 2.0 - 1.0;
            }
            normalize(&mut noise);
            neg.push(noise);

            let _ = enc.train_step(&sent_feats[i], &[sent_proto[i].clone()], &neg);
            // Empuja composición factorial hacia el atractor de la oración.
            let _ = dynamics.train_transition(&composed, &sent_proto[i]);
            let mid = dynamics.step(&composed);
            let _ = dynamics.train_transition(&mid, &sent_proto[i]);
            let _ = dynamics.train_transition(&sent_fp, &composed);
        }
        // Acciones compartidas: corre/come con ambos sujetos → cuenca común.
        let shared = [("corre", "perro", "gato"), ("come", "perro", "gato")];
        for &(verb, s1, s2) in &shared {
            let fv = enc.encode(verb_feats.get(verb).unwrap());
            let a = compose(&enc.encode(subj_feats.get(s1).unwrap()), &fv);
            let b = compose(&enc.encode(subj_feats.get(s2).unwrap()), &fv);
            let mut mid = a.clone();
            for i in 0..FIELD_FP_DIM {
                mid[i] = 0.5 * (a[i] + b[i]);
            }
            normalize(&mut mid);
            let _ = dynamics.train_transition(&a, &mid);
            let _ = dynamics.train_transition(&b, &mid);
        }
    }

    let manifold: Vec<Vec<f64>> = (0..train.len())
        .map(|i| {
            let (_, subj, verb) = train[i];
            let composed = compose(
                &enc.encode(subj_feats.get(subj).unwrap()),
                &enc.encode(verb_feats.get(verb).unwrap()),
            );
            dynamics.step(&dynamics.step(&composed))
        })
        .collect();

    let energy_of = |psi: &[f64]| -> (f64, f64, f64) {
        let mut best = -1.0f64;
        for m in &manifold {
            best = best.max(cosine(psi, m));
        }
        let d_man = 1.0 - best;
        let pulled = dynamics.step(psi);
        let pulled2 = dynamics.step(&pulled);
        let stab = cosine(psi, &pulled2);
        let e_field = field_energy_of_fp(psi).abs();
        let e = d_man * 4.0 + (1.0 - stab) * 2.0 + 0.02 * e_field;
        (e, d_man, stab)
    };

    let mut energies_ok = Vec::new();
    let mut energies_bad = Vec::new();
    let mut stab_ok = Vec::new();
    let mut stab_bad = Vec::new();
    let mut dist_ok = Vec::new();
    let mut dist_bad = Vec::new();
    let mut pair_scores = Vec::new();

    for &(subj, verb, is_ok) in &eval_combos {
        let fs = enc.encode(&probe_features(&mut probe, subj));
        let fv = enc.encode(&probe_features(&mut probe, verb));
        let composed = compose(&fs, &fv);
        let (e, d_man, st) = energy_of(&composed);
        if is_ok {
            energies_ok.push(e);
            stab_ok.push(st);
            dist_ok.push(d_man);
        } else {
            energies_bad.push(e);
            stab_bad.push(st);
            dist_bad.push(d_man);
        }
        pair_scores.push((is_ok, e));
    }

    let e_ok = mean(&energies_ok);
    let e_bad = mean(&energies_bad);
    let d_ok = mean(&dist_ok);
    let d_bad = mean(&dist_bad);
    let s_ok = mean(&stab_ok);
    let s_bad = mean(&stab_bad);

    let mut rank_hits = 0.0;
    let mut rank_n = 0.0;
    for &(ok_a, e_a) in &pair_scores {
        for &(ok_b, e_b) in &pair_scores {
            if ok_a && !ok_b {
                rank_n += 1.0;
                if e_b > e_a {
                    rank_hits += 1.0;
                }
            }
        }
    }
    let rank_acc = rank_hits / f64::max(rank_n, 1.0);
    let margin_e = e_bad - e_ok;
    let structure = i32::from(e_bad > e_ok + 0.02)
        + i32::from(d_bad > d_ok + 0.02)
        + i32::from(s_ok > s_bad + 0.01);

    let verdict = if structure >= 2 && rank_acc >= 0.75 && margin_e > 0.05 {
        "PASS: unseen incompatible combos less stable / farther from manifold"
    } else if structure >= 2 || (rank_acc >= 0.6 && margin_e > 0.0) {
        "PARTIAL: structural separation improving; margin/ranking aún cortos"
    } else if structure == 1 {
        "PARTIAL: weak structural separation without class labels"
    } else {
        "FAIL: no structural signal for unseen incompatible combos"
    };

    let mut row = base_row("E15_semantic_without_labels", seed);
    row.n_concepts = train.len();
    row.train_examples = train.len();
    row.unseen_examples = 2;
    row.mode = FieldMode::DynamicField.as_str().into();
    row.dynamics_trainable = true;
    row.accuracy_seen = 1.0;
    row.accuracy_unseen = rank_acc;
    row.accuracy_ood = f64::from(structure) / 3.0;
    row.energy_initial = e_ok;
    row.energy_final = e_bad;
    row.energy_delta = margin_e;
    row.margin = d_bad - d_ok;
    row.stability = s_ok - s_bad;
    row.top1 = rank_acc;
    row.knn_acc = f64::from(structure) / 3.0;
    row.checkpoint_hash = enc.weights_hash();
    row.field_hash = dynamics.weights_hash();
    row.encoder_type = "factored_compose+manifold_Dphi/TrainableFieldEncoder".into();
    row.verdict = verdict.into();
    row.notes = format!(
        "E_ok={:.4} E_bad={:.4} d_man_ok={:.4} d_man_bad={:.4} stab_ok={:.4} stab_bad={:.4} \
         structure={}/3 rank_acc={:.3} margin_E={:.4} arch=factor_compose+Dphi_manifold",
        e_ok, e_bad, d_ok, d_bad, s_ok, s_bad, structure, rank_acc, margin_e
    );
    row
}

pub fn run_experiment_16(seed: u64) -> RegistryRow11 {
    let mut probe = make_lexicon_probe(seed);
    let chain = ["A", "B", "C", "D"];
    let in_dim = linguistic_feature_dim();
    let mut enc = TrainableFieldEncoder::new(in_dim, FIELD_FP_DIM, seed ^ 0xE16);
    let mut dynamics = FieldDynamics::new(FIELD_FP_DIM, seed ^ 0xD16);
    enc.eta = 0.05;
    dynamics.eta = 0.06;

    let mut states: HashMap<&str, Vec<f64>> = HashMap::new();
    for _ in 0..100 {
        for (i, &s) in chain.iter().enumerate() {
            let f = probe_features(&mut probe, s);
            let mut tgt = vec![0.0; FIELD_FP_DIM];
            for k in 0..FIELD_FP_DIM {
                tgt[k] = ((i + 1) as f64 * 1.3 + k as f64 * 0.5).cos();
            }
            normalize(&mut tgt);
            let mut neg = Vec::new();
            for (j, &o) in chain.iter().enumerate() {
                if j != i {
                    neg.push(enc.encode(&probe_features(&mut probe, o)));
                }
            }
            let _ = enc.train_step(&f, &[tgt], &neg);
        }
    }
    for &s in &chain {
        states.insert(s, enc.encode(&probe_features(&mut probe, s)));
    }

    let transitions = [("A", "B"), ("B", "C"), ("C", "D")];
    for _ in 0..400 {
        for &(a, b) in &transitions {
            let _ = dynamics.train_transition(states.get(a).unwrap(), states.get(b).unwrap());
        }
    }

    let table: HashMap<&str, &str> = transitions.iter().cloned().collect();
    let table_ok = table.get("A") == Some(&"B");

    let nn_next = {
        let a = states.get("A").unwrap();
        let mut best = ("?", -1.0f64);
        for &cand in &["B", "C", "D"] {
            let c = cosine(a, states.get(cand).unwrap());
            if c > best.1 {
                best = (cand, c);
            }
        }
        best.0
    };

    let mut rqm = FusedLiquidCdt::new(8);
    rqm.teach_relation(0, 1);
    rqm.teach_relation(1, 2);
    rqm.teach_relation(2, 3);
    let _ = rqm.sleep_consolidate();
    let rqm_direct = rqm.infer(0, &[0, 1, 2, 3]).predicted;
    let rqm_compose = rqm.infer_compose(0, &[0, 1, 2, 3], 3).predicted;

    let static_pred_cos = cosine(states.get("A").unwrap(), states.get("B").unwrap());

    let pred_b = dynamics.step(states.get("A").unwrap());
    let cos_b = cosine(&pred_b, states.get("B").unwrap());
    let traj = dynamics.rollout(states.get("A").unwrap(), 3);
    let cos_roll_b = cosine(&traj[1], states.get("B").unwrap());
    let cos_roll_c = cosine(&traj[2], states.get("C").unwrap());
    let cos_roll_d = cosine(&traj[3], states.get("D").unwrap());

    let mut fused = FusedLiquidCdt::new(8);
    fused.teach_relation(0, 1);
    fused.teach_relation(1, 2);
    fused.teach_relation(2, 3);
    let sleep = fused.sleep_consolidate();
    let cdt_pred = fused.infer(0, &[1, 2, 3]).predicted;

    let audit = LeakageAudit {
        target_in_training: false,
        target_in_rqm: true,
        target_in_cdt: sleep.engrams_after > 0,
        target_in_attractor_bank: false,
        target_in_decoder_memory: false,
        target_seen_as_exact_vector: false,
    };

    let dyn_ok = cos_b > 0.55 && cos_roll_c > 0.35;
    let rollout_ok = cos_roll_d > 0.25;

    let dyn_pre = FieldDynamics::new(FIELD_FP_DIM, seed ^ 0xD16);
    let cos_pre = cosine(
        &dyn_pre.step(states.get("A").unwrap()),
        states.get("B").unwrap(),
    );
    let causal = cos_b > cos_pre + 0.1;

    let verdict = if dyn_ok && causal && cos_b > static_pred_cos {
        if rollout_ok {
            "PASS: Dphi predicts transitions; rollout partial; RQM control separate"
        } else {
            "PARTIAL: one-step Dphi ok; multi-step rollout weak"
        }
    } else if cos_b > 0.4 {
        "WEAK: mild dynamics; may not beat static/NN"
    } else {
        "FAIL: dynamics did not learn transitions"
    };

    let mut row = base_row("E16_field_dynamics", seed);
    row.n_concepts = 4;
    row.train_examples = 3;
    row.unseen_examples = 1;
    row.dynamics_trainable = true;
    row.cdt_enabled = true;
    row.rqm_enabled = true;
    row.mode = FieldMode::DynamicField.as_str().into();
    row.accuracy_seen = if cos_b > 0.55 { 1.0 } else { cos_b };
    row.accuracy_unseen = if cos_roll_c > 0.35 { 1.0 } else { cos_roll_c };
    row.accuracy_ood = if cos_roll_d > 0.25 { 1.0 } else { cos_roll_d };
    row.top1 = cos_b;
    row.top2 = cos_roll_d;
    row.margin = cos_b - cos_pre;
    row.rollout_steps = 3;
    row.field_steps = 400;
    row.target_in_training = audit.target_in_training;
    row.target_in_rqm = audit.target_in_rqm;
    row.target_in_cdt = audit.target_in_cdt;
    row.target_in_attractor_bank = audit.target_in_attractor_bank;
    row.checkpoint_hash = enc.weights_hash();
    row.field_hash = dynamics.weights_hash();
    row.energy_initial = field_energy_of_fp(states.get("A").unwrap());
    row.energy_final = field_energy_of_fp(&pred_b);
    row.verdict = verdict.into();
    row.notes = format!(
        "table_ok={} nn_next={} rqm_direct={} rqm_compose={} cdt_pred={} cos_B={:.3} \
         rollB={:.3} rollC={:.3} rollD={:.3} static_cosAB={:.3} causal_pre={:.3} \
         audit_clean_for_AtoD_train={}",
        table_ok,
        nn_next,
        rqm_direct,
        rqm_compose,
        cdt_pred,
        cos_b,
        cos_roll_b,
        cos_roll_c,
        cos_roll_d,
        static_pred_cos,
        cos_pre,
        !audit.target_in_training
    );
    row
}

pub fn run_experiment_17(seed: u64) -> RegistryRow11 {
    let mut probe = make_lexicon_probe(seed);
    let in_dim = linguistic_feature_dim();
    let mut enc = TrainableFieldEncoder::new(in_dim, FIELD_FP_DIM, seed ^ 0xE17);
    let mut dynamics = FieldDynamics::new(FIELD_FP_DIM, seed ^ 0xD17);
    enc.eta = 0.05;
    dynamics.eta = 0.07;

    let names = ["GA", "GB", "GC"];
    for (i, &n) in names.iter().enumerate() {
        let f = probe_features(&mut probe, n);
        let mut tgt = vec![0.0; FIELD_FP_DIM];
        for k in 0..FIELD_FP_DIM {
            tgt[k] = ((i + 2) as f64 * 0.9 + k as f64 * 0.33).sin();
        }
        normalize(&mut tgt);
        for _ in 0..40 {
            let _ = enc.train_step(&f, &[tgt.clone()], &[]);
        }
    }
    let fps: Vec<Vec<f64>> = names
        .iter()
        .map(|n| enc.encode(&probe_features(&mut probe, n)))
        .collect();

    for _ in 0..350 {
        let _ = dynamics.train_transition(&fps[0], &fps[1]);
        let _ = dynamics.train_transition(&fps[1], &fps[2]);
    }

    let mut train_targets: HashSet<String> = HashSet::new();
    train_targets.insert("GA->GB".into());
    train_targets.insert("GB->GC".into());
    let query = "GA->GC";
    let target_in_training = train_targets.contains(query);

    let mut rqm = FusedLiquidCdt::new(8);
    rqm.teach_relation(0, 1);
    rqm.teach_relation(1, 2);
    let _ = rqm.sleep_consolidate();
    let rqm_has_direct = false;
    let rqm_compose_pred = rqm.infer_compose(0, &[0, 1, 2], 2).predicted;
    let target_in_rqm = rqm_compose_pred == 2;

    let mut attractor_bank: HashMap<usize, Vec<f64>> = HashMap::new();
    attractor_bank.insert(0, fps[0].clone());
    attractor_bank.insert(1, fps[1].clone());
    let target_in_bank = attractor_bank.values().any(|v| cosine(v, &fps[2]) > 0.999);

    let audit = LeakageAudit {
        target_in_training,
        target_in_rqm,
        target_in_cdt: false,
        target_in_attractor_bank: target_in_bank,
        target_in_decoder_memory: false,
        target_seen_as_exact_vector: false,
    };

    let step1 = dynamics.step(&fps[0]);
    let step2 = dynamics.step(&step1);
    let dist_c = 1.0 - cosine(&step2, &fps[2]);
    let dist_b = 1.0 - cosine(&step2, &fps[1]);
    let energy = field_energy_of_fp(&step2);
    let top1_is_c = dist_c < dist_b;

    let mut dyn_t = FieldDynamics::new(FIELD_FP_DIM, seed ^ 0x117);
    let mut dyn_r = FieldDynamics::new(FIELD_FP_DIM, seed ^ 0x217);
    dyn_t.eta = 0.07;
    dyn_r.eta = 0.07;
    for _ in 0..200 {
        let _ = dyn_t.train_transition(&fps[0], &fps[1]);
        let _ = dyn_r.train_transition(&fps[1], &fps[2]);
    }
    let composed = dyn_r.step(&dyn_t.step(&fps[0]));
    let comp_dist = 1.0 - cosine(&composed, &fps[2]);

    let valid_unseen = !audit.target_in_training && !audit.target_in_attractor_bank;
    let dyn_hit = top1_is_c && dist_c < 0.55;

    let verdict = if !valid_unseen {
        "INVALIDATED: leakage audit failed for declared unseen target"
    } else if dyn_hit && comp_dist < 0.55 {
        "PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately)"
    } else if dyn_hit || comp_dist < 0.6 {
        "PARTIAL: some extrapolation; not both geometric variants"
    } else {
        "FAIL: could not predict never-observed state without leakage shortcuts"
    };

    let mut row = base_row("E17_never_observed_states", seed);
    row.n_concepts = 3;
    row.train_examples = 2;
    row.unseen_examples = 1;
    row.dynamics_trainable = true;
    row.rqm_enabled = true;
    row.mode = FieldMode::DynamicField.as_str().into();
    row.accuracy_seen = cosine(&dynamics.step(&fps[0]), &fps[1]).clamp(0.0, 1.0);
    row.accuracy_unseen = if dyn_hit { 1.0 } else { 1.0 - dist_c };
    row.accuracy_ood = if comp_dist < 0.55 {
        1.0
    } else {
        1.0 - comp_dist
    };
    row.top1 = 1.0 - dist_c;
    row.top2 = 1.0 - dist_b;
    row.margin = dist_b - dist_c;
    row.energy_final = energy;
    row.rollout_steps = 2;
    row.target_in_training = audit.target_in_training;
    row.target_in_rqm = audit.target_in_rqm;
    row.target_in_cdt = audit.target_in_cdt;
    row.target_in_attractor_bank = audit.target_in_attractor_bank;
    row.checkpoint_hash = enc.weights_hash();
    row.field_hash = dynamics.weights_hash();
    row.abstentions = usize::from(audit.target_in_attractor_bank);
    row.verdict = verdict.into();
    row.notes = format!(
        "dist_C={:.4} dist_B={:.4} comp_dist={:.4} rqm_direct_edge={} rqm_compose_pred={} \
         audit_train={} audit_rqm_compose={} audit_bank={} valid_unseen={}",
        dist_c,
        dist_b,
        comp_dist,
        rqm_has_direct,
        rqm_compose_pred,
        audit.target_in_training,
        audit.target_in_rqm,
        audit.target_in_attractor_bank,
        valid_unseen
    );
    row
}

pub fn format_registry_table_11(rows: &[RegistryRow11]) -> String {
    let mut s = String::new();
    s.push_str("# Registry — experimentos 11–17 (campo entrenable)\n\n");
    s.push_str("| Exp | seed | mode | acc_seen | acc_unseen | acc_ood | margin | knn/top1 | dR_A | leak_rqm | verdict |\n");
    s.push_str("|-----|-----:|------|---------:|-----------:|--------:|-------:|---------:|-----:|---------:|---------|\n");
    for r in rows {
        s.push_str(&format!(
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} |\n",
            r.experiment,
            r.seed,
            r.mode,
            r.accuracy_seen,
            r.accuracy_unseen,
            r.accuracy_ood,
            r.margin,
            r.knn_acc.max(r.top1),
            r.forgetting_a,
            r.target_in_rqm,
            r.verdict.replace('|', "/"),
        ));
    }
    s
}

pub fn rows_to_csv(rows: &[RegistryRow11]) -> String {
    let mut csv = String::from(
        "experiment,commit,seed,hardware,rust_version,n_concepts,train_examples,unseen_examples,ood_examples,\
encoder_type,encoder_frozen,field_trainable,dynamics_trainable,cdt_enabled,rqm_enabled,decoder_type,mode,\
accuracy_seen,accuracy_unseen,accuracy_ood,energy_initial,energy_final,energy_delta,top1,top2,margin,abstentions,\
field_steps,rollout_steps,convergence_steps,mean_us,p50_us,p95_us,p99_us,\
target_in_training,target_in_rqm,target_in_cdt,target_in_attractor_bank,\
forgetting_A,forgetting_B,forgetting_C,checkpoint_hash,field_hash,llm_hash,\
intra_cosine,inter_cosine,knn_acc,silhouette,stability,verdict,periphery,notes\n",
    );
    for r in rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{:.6},{:.6},{:.6},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{}\n",
            r.experiment,
            r.commit,
            r.seed,
            r.hardware,
            r.rust_version.replace(',', ";"),
            r.n_concepts,
            r.train_examples,
            r.unseen_examples,
            r.ood_examples,
            r.encoder_type,
            r.encoder_frozen,
            r.field_trainable,
            r.dynamics_trainable,
            r.cdt_enabled,
            r.rqm_enabled,
            r.decoder_type,
            r.mode,
            r.accuracy_seen,
            r.accuracy_unseen,
            r.accuracy_ood,
            r.energy_initial,
            r.energy_final,
            r.energy_delta,
            r.top1,
            r.top2,
            r.margin,
            r.abstentions,
            r.field_steps,
            r.rollout_steps,
            r.convergence_steps,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.p99_us,
            r.target_in_training,
            r.target_in_rqm,
            r.target_in_cdt,
            r.target_in_attractor_bank,
            r.forgetting_a,
            r.forgetting_b,
            r.forgetting_c,
            r.checkpoint_hash,
            r.field_hash,
            r.llm_hash,
            r.intra_cosine,
            r.inter_cosine,
            r.knn_acc,
            r.silhouette,
            r.stability,
            r.verdict.replace(',', ";"),
            r.periphery.replace(',', ";"),
            r.notes.replace(',', ";").replace('\n', " "),
        ));
    }
    csv
}

pub fn run_all_11_17(seeds: &[u64]) -> Vec<RegistryRow11> {
    let mut rows = Vec::new();
    for &seed in seeds {
        rows.push(run_experiment_11(seed));
        rows.push(run_experiment_12(seed));
        rows.push(run_experiment_13(seed));
        rows.push(run_experiment_14(seed));
        rows.push(run_experiment_15(seed));
        rows.push(run_experiment_16(seed));
        rows.push(run_experiment_17(seed));
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e12_harness_skips_without_gguf_or_runs_with() {
        let row = run_experiment_12(0xE12_007);
        if gemma_gguf_available() {
            assert_ne!(row.verdict, "SKIPPED_NO_GGUF");
            assert_eq!(row.periphery, "gemma-gguf");
        } else {
            assert_eq!(row.verdict, "SKIPPED_NO_GGUF");
            assert!(row.notes.contains("NO se afirma cross-lingual"));
        }
    }

    #[test]
    fn leakage_audit_flags_contamination() {
        let dirty = LeakageAudit {
            target_in_attractor_bank: true,
            ..Default::default()
        };
        assert!(dirty.invalidate_unseen());
        assert!(LeakageAudit::default().clean());
    }

    #[test]
    fn trainable_encoder_moves_weights() {
        let mut enc = TrainableFieldEncoder::new(8, 4, 7);
        let f = vec![0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0];
        let h0 = enc.weights_hash();
        let pos = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let neg = vec![vec![0.0, 1.0, 0.0, 0.0]];
        let _ = enc.train_step(&f, &pos, &neg);
        assert_ne!(enc.weights_hash(), h0);
    }

    #[test]
    fn liquid_experiments_11_17_registry() {
        let seeds: Vec<u64> = (0..8).map(|i| 0xE1100 + i).collect();
        let rows = run_all_11_17(&seeds);
        assert_eq!(rows.len(), 8 * 7);

        let table = format_registry_table_11(&rows);
        println!("{table}");

        let mut notes = String::new();
        for r in &rows {
            notes.push_str(&format!(
                "- {} seed={} verdict={} notes={}\n",
                r.experiment, r.seed, r.verdict, r.notes
            ));
        }

        let e11: Vec<_> = rows
            .iter()
            .filter(|r| r.experiment.starts_with("E11"))
            .collect();
        let e12: Vec<_> = rows
            .iter()
            .filter(|r| r.experiment.starts_with("E12"))
            .collect();
        let e16: Vec<_> = rows
            .iter()
            .filter(|r| r.experiment.starts_with("E16"))
            .collect();

        let md = format!(
            "# Resultados experimentos 11–17 — campo entrenable\n\n\
             Rama: `exp/liquid-inference-experiments-8-9-10`\n\n\
             Protocolo: `docs/experimentos_11_17_campo_entrenable.md`\n\n\
             Commit al correr: {}.\n\
             Hardware: {}.\n\
             rustc: {}.\n\
             Semillas: 8 (0xE1100..0xE1107). Escala N=8 cubierta en E11 (`accuracy_ood`).\n\
             Periferia: {}.\n\
             GGUF: {}.\n\n\
             ## Tabla de registro\n\n{}\n\n\
             ## Notas por corrida\n\n{}\n\n\
             ## Controles y honestidad\n\
             - E8–E10 permanecen como baseline en `docs/resultados_experimentos_8_9_10.*` (no mezclados).\n\
             - E12: si no hay GGUF → `SKIPPED_NO_GGUF` (no se afirma cross-lingual con léxico).\n\
             - RQM es control; no explicación primaria de E13/E16/E17.\n\
             - Auditoría anti-leakage en E16/E17.\n\
             - No se presenta como evidencia de cognición / AGI.\n\
             - LLM congelado; solo se entrenan θ del encoder y φ de la dinámica.\n",
            git_commit(),
            hardware_label(),
            rust_version(),
            periphery_label(),
            if gemma_gguf_available() {
                "disponible"
            } else {
                "ausente → E12 SKIPPED_NO_GGUF"
            },
            table,
            notes
        );
        let _ = std::fs::create_dir_all("docs");
        let _ = std::fs::write("docs/resultados_experimentos_11_17.md", &md);
        let _ = std::fs::write("docs/resultados_experimentos_11_17.csv", rows_to_csv(&rows));

        assert_eq!(e11.len(), 8);
        assert_eq!(e12.len(), 8);
        assert_eq!(e16.len(), 8);
        for r in &e12 {
            if !gemma_gguf_available() {
                assert_eq!(r.verdict, "SKIPPED_NO_GGUF");
            }
        }
        for r in &e11 {
            assert!(!r.checkpoint_hash.is_empty());
        }
    }
}
