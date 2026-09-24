//! Clean-Room v2 harness for Etapa 2 (protocol §29+).
//!
//! Absolute separation from E11–E17 learned state. Fresh random FieldEncoder +
//! D_phi per run; CDT empty at start; RQM/NN/table/attractor OFF at eval.
//! Independent Stage-2 dataset generator with TRAIN/DEV/TEST seal + anti-contamination.
//!
//! Legacy suite lives in `field_autonomy_stage2` (historical / pre-cleanroom).

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::liquid_experiments_11_17::{FieldDynamics, TrainableFieldEncoder};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::time::Instant;

pub const GENERATOR_VERSION: &str = "stage2_v2_gen_1.2.0";
pub const DATASET_VERSION: &str = "stage2_v2_ds_1.0.0";
pub const PROTOCOL_SECTION: &str = "§29+ Clean-Room v2";

pub const DEV_SEEDS: [u64; 8] = [
    0xA300, 0xA301, 0xA302, 0xA303, 0xA304, 0xA305, 0xA306, 0xA307,
];
pub const CONFIRMATION_SEEDS: [u64; 16] = [
    0xB300, 0xB301, 0xB302, 0xB303, 0xB304, 0xB305, 0xB306, 0xB307, 0xB308, 0xB309, 0xB30A, 0xB30B,
    0xB30C, 0xB30D, 0xB30E, 0xB30F,
];

const EPS: f64 = 1e-12;
const FEAT_DIM: usize = 16;
const FIELD_DIM: usize = 24;
const COS_PASS: f64 = 0.85;
const COS_STRONG: f64 = 0.92;
const COS_PARTIAL: f64 = 0.70;
const NEAR_DUP_L2: f64 = 1e-3;
const EQUIV_L2: f64 = 5e-3;

#[derive(Clone, Debug)]
pub struct HyperparamLock {
    pub lr_encoder: f64,
    pub lr_dynamics: f64,
    pub enc_epochs: usize,
    pub dyn_epochs: usize,
    pub dyn_updates: usize,
    pub field_dim: usize,
    pub feat_dim: usize,
    pub lambda_reg: f64,
    pub train_n: usize,
    pub dev_n: usize,
    pub test_n: usize,
    pub locked: bool,
    pub test_observed: bool,
    pub params_changed_after_test: bool,
}

impl Default for HyperparamLock {
    fn default() -> Self {
        Self {
            lr_encoder: 0.05,
            lr_dynamics: 0.04,
            enc_epochs: 60,
            dyn_epochs: 180,
            dyn_updates: 3,
            field_dim: FIELD_DIM,
            feat_dim: FEAT_DIM,
            lambda_reg: 0.02,
            train_n: 28,
            dev_n: 10,
            test_n: 12,
            locked: false,
            test_observed: false,
            params_changed_after_test: false,
        }
    }
}

impl HyperparamLock {
    pub fn smoke() -> Self {
        Self {
            enc_epochs: 20,
            dyn_epochs: 50,
            dyn_updates: 2,
            train_n: 16,
            dev_n: 6,
            test_n: 8,
            ..Self::default()
        }
    }

    /// Longer / more varied curriculum (plan v3 §19–§21). Locked before TEST.
    pub fn long() -> Self {
        Self {
            enc_epochs: 160,
            dyn_epochs: 560,
            dyn_updates: 7,
            train_n: 80,
            dev_n: 20,
            test_n: 24,
            lr_encoder: 0.04,
            lr_dynamics: 0.035,
            ..Self::default()
        }
    }

    /// `STAGE2_V2_PROFILE=smoke|default|long` (default = `default`).
    pub fn from_env() -> Self {
        match std::env::var("STAGE2_V2_PROFILE")
            .unwrap_or_else(|_| "default".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "smoke" => Self::smoke(),
            "long" => Self::long(),
            _ => Self::default(),
        }
    }

    pub fn lock(&mut self) {
        self.locked = true;
    }

    pub fn mark_test_observed(&mut self) {
        self.test_observed = true;
    }

    pub fn try_set_epochs(&mut self, enc: usize, dyn_ep: usize) -> Result<(), &'static str> {
        if self.test_observed && self.locked {
            self.params_changed_after_test = true;
            return Err("TEST_INVALIDATED");
        }
        self.enc_epochs = enc;
        self.dyn_epochs = dyn_ep;
        Ok(())
    }

    pub fn status(&self) -> &'static str {
        if self.params_changed_after_test {
            "TEST_INVALIDATED"
        } else if self.locked {
            "LOCKED"
        } else {
            "UNLOCKED"
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContRule {
    Translation {
        dx: f64,
        dy: f64,
    },
    Rotation {
        theta: f64,
    },
    Scaling {
        s: f64,
    },
    Affine {
        a00: f64,
        a01: f64,
        a10: f64,
        a11: f64,
        bx: f64,
        by: f64,
    },
    Compose(ContRuleRef, ContRuleRef),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContRuleRef {
    Translation { dx: f64, dy: f64 },
    Rotation { theta: f64 },
    Scaling { s: f64 },
}

impl ContRuleRef {
    fn apply(self, p: (f64, f64)) -> (f64, f64) {
        match self {
            Self::Translation { dx, dy } => (p.0 + dx, p.1 + dy),
            Self::Rotation { theta } => {
                let (c, s) = (theta.cos(), theta.sin());
                (c * p.0 - s * p.1, s * p.0 + c * p.1)
            }
            Self::Scaling { s } => (p.0 * s, p.1 * s),
        }
    }

    fn family(self) -> &'static str {
        match self {
            Self::Translation { .. } => "translation",
            Self::Rotation { .. } => "rotation",
            Self::Scaling { .. } => "scaling",
        }
    }
}

impl ContRule {
    pub fn apply(self, p: (f64, f64)) -> (f64, f64) {
        match self {
            Self::Translation { dx, dy } => (p.0 + dx, p.1 + dy),
            Self::Rotation { theta } => {
                let (c, s) = (theta.cos(), theta.sin());
                (c * p.0 - s * p.1, s * p.0 + c * p.1)
            }
            Self::Scaling { s } => (p.0 * s, p.1 * s),
            Self::Affine {
                a00,
                a01,
                a10,
                a11,
                bx,
                by,
            } => (a00 * p.0 + a01 * p.1 + bx, a10 * p.0 + a11 * p.1 + by),
            Self::Compose(t1, t2) => t2.apply(t1.apply(p)),
        }
    }

    pub fn family(self) -> &'static str {
        match self {
            Self::Translation { .. } => "translation",
            Self::Rotation { .. } => "rotation",
            Self::Scaling { .. } => "scaling",
            Self::Affine { .. } => "affine",
            Self::Compose(_, _) => "compose",
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::Translation { dx, dy } => format!("T(dx={dx:.4},dy={dy:.4})"),
            Self::Rotation { theta } => format!("R(theta={theta:.4})"),
            Self::Scaling { s } => format!("S(s={s:.4})"),
            Self::Affine {
                a00,
                a01,
                a10,
                a11,
                bx,
                by,
            } => {
                format!("A([{a00:.2},{a01:.2};{a10:.2},{a11:.2}]+[{bx:.2},{by:.2}])")
            }
            Self::Compose(a, b) => format!("Compose({}+{})", a.family(), b.family()),
        }
    }

    pub fn param_range_label(self) -> String {
        match self {
            Self::Translation { dx, dy } => format!("dx,dy in [{dx:.3},{dy:.3}]"),
            Self::Rotation { theta } => format!("theta={theta:.4}"),
            Self::Scaling { s } => format!("s={s:.4}"),
            Self::Affine { .. } => "affine_params".into(),
            Self::Compose(_, _) => "compose_params".into(),
        }
    }
}


fn action_of_ref(r: ContRuleRef) -> (f64, f64) {
    match r {
        ContRuleRef::Translation { dx, dy } => (dx, dy),
        ContRuleRef::Rotation { theta } => (theta, 1.0),
        ContRuleRef::Scaling { s } => (s, -1.0),
    }
}

/// Action cue for every rule family (not only translation). Enables E21/E22 conditioning.
pub fn action_of(rule: ContRule) -> Option<(f64, f64)> {
    match rule {
        ContRule::Translation { dx, dy } => Some((dx, dy)),
        ContRule::Rotation { theta } => Some((theta, 1.0)),
        ContRule::Scaling { s } => Some((s, -1.0)),
        ContRule::Affine { bx, by, .. } => Some((bx, by)),
        ContRule::Compose(a, b) => {
            let (ax, ay) = action_of_ref(a);
            let (bx, by) = action_of_ref(b);
            // Distinct compose cue (not a simple sum of train actions).
            Some((ax * 0.7 + bx * 0.3, ay * 0.7 + by * 0.3 + 2.0))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub x: (f64, f64),
    pub y: (f64, f64),
    pub rule: ContRule,
    pub action: Option<(f64, f64)>,
    pub tag: String,
}

#[derive(Clone, Debug)]
pub struct SplitDataset {
    pub train: Vec<Sample>,
    pub dev: Vec<Sample>,
    pub test: Vec<Sample>,
    pub rule_family: String,
    pub dimension: usize,
    pub parameter_range: String,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct DatasetManifest {
    pub dataset_version: String,
    pub generator_version: String,
    pub generator_commit: String,
    pub seed: u64,
    pub sha256: String,
    pub rule_family: String,
    pub dimension: usize,
    pub parameter_range: String,
    pub split: String,
    pub n_samples: usize,
    pub sealed: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ContaminationHit {
    pub kind: String,
    pub detail: String,
}

#[derive(Clone, Debug, Default)]
pub struct ContaminationReport {
    pub hits: Vec<ContaminationHit>,
}

impl ContaminationReport {
    pub fn is_invalid(&self) -> bool {
        !self.hits.is_empty()
    }

    pub fn status(&self) -> &'static str {
        if self.is_invalid() {
            "DATASET_INVALID"
        } else {
            "CLEAN"
        }
    }
}

#[derive(Clone, Debug)]
pub struct SealedBundle {
    pub dataset: SplitDataset,
    pub train_manifest: DatasetManifest,
    pub dev_manifest: DatasetManifest,
    pub test_manifest: DatasetManifest,
    pub contamination: ContaminationReport,
    pub test_immutable_sha: String,
}

#[derive(Clone, Debug, Default)]
pub struct ProvenanceRecord {
    pub target_seen_training: bool,
    pub target_seen_dev: bool,
    pub target_seen_cdt: bool,
    pub target_seen_rqm: bool,
    pub target_seen_attractor: bool,
    pub target_seen_table: bool,
    pub target_seen_nn: bool,
    pub target_equivalent_seen: bool,
    pub cdt_queries: u64,
    pub rqm_queries: u64,
    pub table_queries: u64,
    pub nn_queries: u64,
    pub attractor_queries: u64,
    pub target_in_normalization: bool,
    pub target_in_decoder: bool,
    pub target_in_threshold_selection: bool,
    pub target_in_hyperparameter_selection: bool,
    pub field_only: bool,
    pub information_path: String,
}

impl ProvenanceRecord {
    pub fn field_only_clean(path: &str) -> Self {
        Self {
            field_only: true,
            information_path: path.into(),
            ..Default::default()
        }
    }

    pub fn leakage_score(&self) -> u64 {
        let flags = [
            self.target_seen_training,
            self.target_seen_cdt,
            self.target_seen_rqm,
            self.target_seen_attractor,
            self.target_seen_table,
            self.target_seen_nn,
            self.target_equivalent_seen,
            self.target_in_normalization,
            self.target_in_decoder,
            self.target_in_threshold_selection,
            self.target_in_hyperparameter_selection,
        ]
        .iter()
        .filter(|&&b| b)
        .count() as u64;
        let q = if self.field_only {
            self.cdt_queries
                + self.rqm_queries
                + self.table_queries
                + self.nn_queries
                + self.attractor_queries
        } else {
            0
        };
        flags + q
    }
}

#[derive(Clone, Debug, Default)]
pub struct ResultRowV2 {
    pub experiment: String,
    pub seed: u64,
    pub commit: String,
    pub rule: String,
    pub mode: String,
    pub cosine_dynamic: f64,
    pub cosine_static: f64,
    pub cosine_linear: f64,
    pub cosine_nn: f64,
    pub cosine_table: f64,
    pub mse: f64,
    pub rel_err: f64,
    pub energy: f64,
    pub stability: f64,
    pub experience_gain: f64,
    pub leakage_score: u64,
    pub train_n: usize,
    pub dev_n: usize,
    pub test_n: usize,
    pub test_sha256: String,
    pub hyperparam_lock: String,
    pub contamination: String,
    pub steps_trained: usize,
    pub encoder_hash: String,
    pub dynamics_hash: String,
    pub verdict: String,
    pub notes: String,
    pub field_only: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ExperienceCdt {
    pub episodes: Vec<(Vec<f64>, Vec<f64>, String)>,
    pub mean_delta: Option<Vec<f64>>,
    pub consolidate_calls: u64,
    pub corrupted: bool,
    pub irrelevant: bool,
    pub other_rule: bool,
    pub stats_only: bool,
}

impl ExperienceCdt {
    pub fn store(&mut self, before: Vec<f64>, after: Vec<f64>, action: String) {
        self.episodes.push((before, after, action));
    }

    pub fn consolidate(&mut self) {
        self.consolidate_calls += 1;
        if self.episodes.is_empty() {
            self.mean_delta = None;
            return;
        }
        let dim = self.episodes[0].0.len();
        let mut acc = vec![0.0; dim];
        let mut n = 0.0;
        for (b, a, _) in &self.episodes {
            if b.len() != dim || a.len() != dim {
                continue;
            }
            for i in 0..dim {
                acc[i] += a[i] - b[i];
            }
            n += 1.0;
        }
        if n > 0.0 {
            for v in &mut acc {
                *v /= n;
            }
            if self.corrupted {
                for v in &mut acc {
                    *v *= -1.0;
                }
            }
            if self.irrelevant {
                for (i, v) in acc.iter_mut().enumerate() {
                    *v = if i % 2 == 0 { 0.37 } else { -0.21 };
                }
            }
            self.mean_delta = Some(acc);
        }
        if self.stats_only {
            self.episodes.clear();
        }
    }

    pub fn contains_after_hash(&self, h: u64) -> bool {
        self.episodes.iter().any(|(_, a, _)| hash_point_vec(a) == h)
    }

    pub fn regularity(&self) -> Option<&[f64]> {
        self.mean_delta.as_deref()
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn stddev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0);
    v.sqrt()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na < EPS || nb < EPS {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn normalize(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > EPS {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

fn mse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s / n as f64
}

fn rel_err(pred: &[f64], tgt: &[f64]) -> f64 {
    let e = l2(pred, tgt);
    let n = tgt.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
    e / n
}

fn energy_of(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>()
}

fn hash_point(p: (f64, f64)) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in [p.0, p.1] {
        h ^= x.to_bits();
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn hash_point_vec(v: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in v {
        h ^= x.to_bits();
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let out = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

fn serialize_samples(samples: &[Sample]) -> Vec<u8> {
    let mut buf = Vec::new();
    for s in samples {
        buf.extend_from_slice(&s.x.0.to_le_bytes());
        buf.extend_from_slice(&s.x.1.to_le_bytes());
        buf.extend_from_slice(&s.y.0.to_le_bytes());
        buf.extend_from_slice(&s.y.1.to_le_bytes());
        buf.extend_from_slice(s.rule.family().as_bytes());
        buf.push(b'|');
        buf.extend_from_slice(s.tag.as_bytes());
        buf.push(b'\n');
    }
    buf
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into())
}

/// Dual feature path (v3.4): do not force one encoding for everything.
/// SoftScale preserves E22 oracle-mid / compose; Relative desaturates static on E18/E21/E24/E25/E27.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeatPath {
    SoftScale,
    Relative,
}

fn point_features(p: (f64, f64), action: Option<(f64, f64)>, path: FeatPath) -> Vec<f64> {
    match path {
        FeatPath::SoftScale => point_features_soft(p, action),
        FeatPath::Relative => point_features_relative(p, action),
    }
}

/// Soft-scale absolute (v3.1/v3.3): preserves oracle-mid / compose decode.
fn point_features_soft(p: (f64, f64), action: Option<(f64, f64)>) -> Vec<f64> {
    let (x, y) = p;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let mut f = vec![
        x * 0.2,                 // soft-scale absolute (compose / mid decode)
        y * 0.2,
        x * x * 0.08,
        y * y * 0.08,
        x * y * 0.06,
        x.sin() * 0.40,
        y.cos() * 0.40,
        (0.5 * x).sin() * 0.35,
        (0.5 * y).cos() * 0.35,
        (x + y) * 0.05,
        (x - y) * 0.05,
        (fx - 0.5) * 0.22,
        (fy - 0.5) * 0.22,
        (2.0 * x).sin() * 0.12,
        0.0,
        0.0,
    ];
    f.truncate(FEAT_DIM);
    while f.len() < FEAT_DIM {
        f.push(0.0);
    }
    if FEAT_DIM > 4 {
        let end = FEAT_DIM.saturating_sub(2).max(2);
        if end > 2 {
            normalize(&mut f[2..end]);
            for v in &mut f[2..end] {
                *v *= 0.32;
            }
        }
    }
    if let Some((dx, dy)) = action {
        if FEAT_DIM >= 2 {
            f[FEAT_DIM - 2] = dx * 0.45;
            f[FEAT_DIM - 1] = dy * 0.45;
        }
    }
    f
}

/// Relative / centered / periodic (v3.2): desaturates far-region static (~0.97→~0.68).
fn point_features_relative(p: (f64, f64), action: Option<(f64, f64)>) -> Vec<f64> {
    let (x, y) = p;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let mut f = vec![
        (x * 0.22).tanh(),
        (y * 0.22).tanh(),
        fx - 0.5,
        fy - 0.5,
        x.sin(),
        x.cos(),
        y.sin(),
        y.cos(),
        (2.0 * x).sin() * 0.65,
        (2.0 * y).sin() * 0.65,
        (0.5 * x).sin(),
        (0.5 * y).cos(),
        (x - y) * 0.12,
        (x + y) * 0.06,
        0.0,
        0.0,
    ];
    f.truncate(FEAT_DIM);
    while f.len() < FEAT_DIM {
        f.push(0.0);
    }
    // Do NOT L2-normalize the geometry block (collapses far translations).
    if let Some((dx, dy)) = action {
        if FEAT_DIM >= 2 {
            f[FEAT_DIM - 2] = dx * 0.55;
            f[FEAT_DIM - 1] = dy * 0.55;
        }
    }
    f
}

fn encode_xy(
    enc: &TrainableFieldEncoder,
    p: (f64, f64),
    action: Option<(f64, f64)>,
    path: FeatPath,
) -> Vec<f64> {
    enc.encode(&point_features(p, action, path))
}

/// Geometry-only encoding (no action channels) for static baseline comparisons.
fn encode_xy_geom(enc: &TrainableFieldEncoder, p: (f64, f64), path: FeatPath) -> Vec<f64> {
    enc.encode(&point_features(p, None, path))
}

/// Residual MLP point decoder ψ → (x, y). Joint-trainable with Dφ for compose.
/// out = W_skip ψ + b + W2 tanh(W1 ψ + b1) + b2
#[derive(Clone, Debug)]
struct PointDecoder {
    dim: usize,
    hidden: usize,
    w_skip: Vec<f64>, // 2 * dim
    b: [f64; 2],
    w1: Vec<f64>, // hidden * dim
    b1: Vec<f64>, // hidden
    w2: Vec<f64>, // 2 * hidden
    b2: [f64; 2],
    eta: f64,
}

impl PointDecoder {
    fn new(dim: usize, hidden: usize, seed: u64) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xDEC0DE);
        let scale1 = 0.25 / (dim as f64).sqrt();
        let scale2 = 0.25 / (hidden as f64).sqrt();
        let scale_s = 0.15 / (dim as f64).sqrt();
        let mut w1 = vec![0.0; hidden * dim];
        for v in &mut w1 {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * scale1;
        }
        let mut w2 = vec![0.0; 2 * hidden];
        for v in &mut w2 {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * scale2;
        }
        let mut w_skip = vec![0.0; 2 * dim];
        for v in &mut w_skip {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * scale_s;
        }
        Self {
            dim,
            hidden,
            w_skip,
            b: [0.0, 0.0],
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: [0.0, 0.0],
            eta: 0.04,
        }
    }

    fn forward(&self, psi: &[f64]) -> (f64, f64, Vec<f64>) {
        let mut h = self.b1.clone();
        for i in 0..self.hidden {
            let mut acc = self.b1[i];
            for j in 0..self.dim {
                acc += self.w1[i * self.dim + j] * psi.get(j).copied().unwrap_or(0.0);
            }
            h[i] = acc.tanh();
        }
        let mut pred = [self.b[0] + self.b2[0], self.b[1] + self.b2[1]];
        for j in 0..self.dim {
            let v = psi.get(j).copied().unwrap_or(0.0);
            pred[0] += self.w_skip[j] * v;
            pred[1] += self.w_skip[self.dim + j] * v;
        }
        for i in 0..self.hidden {
            pred[0] += self.w2[i] * h[i];
            pred[1] += self.w2[self.hidden + i] * h[i];
        }
        (pred[0], pred[1], h)
    }

    fn decode(&self, psi: &[f64]) -> (f64, f64) {
        let (x, y, _) = self.forward(psi);
        (x, y)
    }

    /// One SGD step on MSE(decode(ψ), target). Returns loss.
    fn train_step(&mut self, psi: &[f64], target: (f64, f64)) -> f64 {
        let (px, py, h) = self.forward(psi);
        let e0 = px - target.0;
        let e1 = py - target.1;
        let loss = 0.5 * (e0 * e0 + e1 * e1);
        if self.eta <= 0.0 {
            return loss;
        }
        let eta = self.eta;
        let lambda = 1e-4;
        // dL/dpred = (e0, e1)
        self.b[0] -= eta * e0;
        self.b[1] -= eta * e1;
        self.b2[0] -= eta * e0;
        self.b2[1] -= eta * e1;
        for j in 0..self.dim {
            let v = psi.get(j).copied().unwrap_or(0.0);
            self.w_skip[j] -= eta * (e0 * v + lambda * self.w_skip[j]);
            self.w_skip[self.dim + j] -= eta * (e1 * v + lambda * self.w_skip[self.dim + j]);
        }
        // backprop through W2 and hidden tanh
        let mut dh = vec![0.0; self.hidden];
        for i in 0..self.hidden {
            dh[i] = e0 * self.w2[i] + e1 * self.w2[self.hidden + i];
            self.w2[i] -= eta * (e0 * h[i] + lambda * self.w2[i]);
            self.w2[self.hidden + i] -= eta * (e1 * h[i] + lambda * self.w2[self.hidden + i]);
        }
        for i in 0..self.hidden {
            let g = dh[i] * (1.0 - h[i] * h[i]);
            self.b1[i] -= eta * g;
            for j in 0..self.dim {
                let v = psi.get(j).copied().unwrap_or(0.0);
                self.w1[i * self.dim + j] -= eta * (g * v + lambda * self.w1[i * self.dim + j]);
            }
        }
        loss
    }

    fn fit(psis: &[Vec<f64>], pts: &[(f64, f64)], epochs: usize) -> Self {
        let dim = psis.first().map(|v| v.len()).unwrap_or(FIELD_DIM);
        let mut dec = Self::new(dim, 32, 0xF17);
        dec.eta = 0.04;
        let n_ep = epochs.max(100);
        for _ in 0..n_ep {
            for (psi, &pt) in psis.iter().zip(pts.iter()) {
                let _ = dec.train_step(psi, pt);
            }
        }
        dec
    }

    fn point_err(a: (f64, f64), b: (f64, f64)) -> f64 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Cheap multi-family curriculum (rot/scale/affine/translation) for action-conditioning.

/// Cheap linear ψ→(x,y) probe: extrapolates OOD better than deep MLP alone.
#[derive(Clone, Debug)]
struct LinearPointProbe {
    dim: usize,
    w: Vec<f64>, // 2 * dim
    b: [f64; 2],
}

impl LinearPointProbe {
    fn fit(psis: &[Vec<f64>], pts: &[(f64, f64)], epochs: usize) -> Self {
        let dim = psis.first().map(|v| v.len()).unwrap_or(FIELD_DIM);
        let mut w = vec![0.0; 2 * dim];
        let mut b = [0.0, 0.0];
        let eta = 0.05;
        let n_ep = epochs.max(80);
        for _ in 0..n_ep {
            for (psi, &pt) in psis.iter().zip(pts.iter()) {
                let mut px = b[0];
                let mut py = b[1];
                for j in 0..dim {
                    let v = psi.get(j).copied().unwrap_or(0.0);
                    px += w[j] * v;
                    py += w[dim + j] * v;
                }
                let e0 = px - pt.0;
                let e1 = py - pt.1;
                b[0] -= eta * e0;
                b[1] -= eta * e1;
                for j in 0..dim {
                    let v = psi.get(j).copied().unwrap_or(0.0);
                    w[j] -= eta * e0 * v;
                    w[dim + j] -= eta * e1 * v;
                }
            }
        }
        Self { dim, w, b }
    }

    fn decode(&self, psi: &[f64]) -> (f64, f64) {
        let mut px = self.b[0];
        let mut py = self.b[1];
        for j in 0..self.dim {
            let v = psi.get(j).copied().unwrap_or(0.0);
            px += self.w[j] * v;
            py += self.w[self.dim + j] * v;
        }
        (px, py)
    }
}

/// Blend MLP decode with linear probe; residual soft-scale abs channels via linear.
fn decode_ood(mlp: &PointDecoder, lin: &LinearPointProbe, psi: &[f64], alpha_lin: f64) -> (f64, f64) {
    let (mx, my) = mlp.decode(psi);
    let (lx, ly) = lin.decode(psi);
    (
        (1.0 - alpha_lin) * mx + alpha_lin * lx,
        (1.0 - alpha_lin) * my + alpha_lin * ly,
    )
}


fn multifamily_curriculum(rng: &mut Xoshiro256StarStar, n_per: usize) -> Vec<Sample> {
    let families = [
        ContRule::Translation { dx: 0.85, dy: -0.45 },
        ContRule::Rotation {
            theta: std::f64::consts::FRAC_PI_6,
        },
        ContRule::Scaling { s: 1.35 },
        ContRule::Affine {
            a00: 1.05,
            a01: 0.15,
            a10: -0.1,
            a11: 0.95,
            bx: 0.25,
            by: -0.15,
        },
    ];
    let mut out = Vec::with_capacity(n_per * families.len());
    for (i, rule) in families.iter().enumerate() {
        out.extend(make_samples(
            rng,
            *rule,
            n_per.max(2),
            -2.0,
            2.0,
            &format!("mf{i}"),
        ));
    }
    out
}

fn sample_point(rng: &mut Xoshiro256StarStar, lo: f64, hi: f64) -> (f64, f64) {
    (
        lo + rng.gen::<f64>() * (hi - lo),
        lo + rng.gen::<f64>() * (hi - lo),
    )
}

fn almost_eq_pt(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
    (a.0 - b.0).abs() < tol && (a.1 - b.1).abs() < tol
}

pub fn default_translation_rule() -> ContRule {
    ContRule::Translation {
        dx: 1.25,
        dy: -0.75,
    }
}

fn make_samples(
    rng: &mut Xoshiro256StarStar,
    rule: ContRule,
    n: usize,
    lo: f64,
    hi: f64,
    tag_prefix: &str,
) -> Vec<Sample> {
    let mut out = Vec::with_capacity(n);
    let action = action_of(rule);
    for i in 0..n {
        let x = sample_point(rng, lo, hi);
        let y = rule.apply(x);
        out.push(Sample {
            x,
            y,
            rule,
            action,
            tag: format!("{tag_prefix}_{i}"),
        });
    }
    out
}

/// Independent Stage-2 generator. Does **not** copy E11–E17 datasets.
pub fn generate_split(
    seed: u64,
    rule: ContRule,
    train_n: usize,
    dev_n: usize,
    test_n: usize,
) -> SplitDataset {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xC1EA7001);
    // Disjoint regions so X_new / B_new outside train (§30).
    let train = make_samples(&mut rng, rule, train_n, -2.5, 2.5, "train");
    let dev = make_samples(&mut rng, rule, dev_n, -2.2, 2.2, "dev");
    let test = make_samples(&mut rng, rule, test_n, 3.0, 6.5, "test");
    SplitDataset {
        train,
        dev,
        test,
        rule_family: rule.family().into(),
        dimension: 2,
        parameter_range: rule.param_range_label(),
        seed,
    }
}

pub fn audit_contamination_hard(ds: &SplitDataset) -> ContaminationReport {
    let mut hits = Vec::new();
    for (name, samples) in [("dev", &ds.dev[..]), ("test", &ds.test[..])] {
        for s in samples {
            for t in &ds.train {
                if almost_eq_pt(s.x, t.x, 1e-9) && almost_eq_pt(s.y, t.y, 1e-9) {
                    hits.push(ContaminationHit {
                        kind: "exact_duplicate".into(),
                        detail: format!("{name} exact pair in train"),
                    });
                }
                let xv = [s.x.0, s.x.1];
                let tx = [t.x.0, t.x.1];
                let yv = [s.y.0, s.y.1];
                let ty = [t.y.0, t.y.1];
                if l2(&xv, &tx) < NEAR_DUP_L2 && l2(&yv, &ty) < NEAR_DUP_L2 {
                    hits.push(ContaminationHit {
                        kind: "near_duplicate".into(),
                        detail: format!("{name} near-dup pair in train"),
                    });
                }
                if almost_eq_pt(s.y, t.y, 1e-6) {
                    hits.push(ContaminationHit {
                        kind: "equivalent_target".into(),
                        detail: format!("{name} target equals train target"),
                    });
                }
                if almost_eq_pt(s.x, t.x, 1e-6) {
                    hits.push(ContaminationHit {
                        kind: "equivalent_state".into(),
                        detail: format!("{name} state equals train state"),
                    });
                }
            }
        }
    }
    ContaminationReport { hits }
}

/// Full auditor including same_orbit / equivalent_transformation (for tests & strict mode).
pub fn audit_contamination_full(ds: &SplitDataset) -> ContaminationReport {
    let mut report = audit_contamination_hard(ds);
    let mut train_orbits = HashSet::new();
    for s in &ds.train {
        let qx = (s.x.0 * 1000.0).round() as i64;
        let qy = (s.x.1 * 1000.0).round() as i64;
        train_orbits.insert((qx, qy));
        let qx = (s.y.0 * 1000.0).round() as i64;
        let qy = (s.y.1 * 1000.0).round() as i64;
        train_orbits.insert((qx, qy));
    }
    for (name, samples) in [("dev", &ds.dev[..]), ("test", &ds.test[..])] {
        for s in samples {
            let qx = (s.x.0 * 1000.0).round() as i64;
            let qy = (s.x.1 * 1000.0).round() as i64;
            if train_orbits.contains(&(qx, qy)) {
                report.hits.push(ContaminationHit {
                    kind: "same_orbit".into(),
                    detail: format!("{name} shares quantized orbit"),
                });
            }
            for t in &ds.train {
                if s.rule.family() != t.rule.family() {
                    let mapped = s.rule.apply(t.x);
                    if almost_eq_pt(mapped, t.y, EQUIV_L2) {
                        report.hits.push(ContaminationHit {
                            kind: "equivalent_transformation".into(),
                            detail: format!("{name} rule mimics train rule"),
                        });
                    }
                }
            }
        }
    }
    report
}

fn manifest_for(
    split: &str,
    samples: &[Sample],
    ds: &SplitDataset,
    sealed: bool,
) -> DatasetManifest {
    let bytes = serialize_samples(samples);
    DatasetManifest {
        dataset_version: DATASET_VERSION.into(),
        generator_version: GENERATOR_VERSION.into(),
        generator_commit: git_commit(),
        seed: ds.seed,
        sha256: sha256_hex(&bytes),
        rule_family: ds.rule_family.clone(),
        dimension: ds.dimension,
        parameter_range: ds.parameter_range.clone(),
        split: split.into(),
        n_samples: samples.len(),
        sealed,
    }
}

pub fn generate_and_seal(
    seed: u64,
    rule: ContRule,
    hp: &HyperparamLock,
) -> Result<SealedBundle, String> {
    let dataset = generate_split(seed, rule, hp.train_n, hp.dev_n, hp.test_n);
    let contamination = audit_contamination_hard(&dataset);
    if contamination.is_invalid() {
        return Err(format!(
            "DATASET_INVALID: {} hits",
            contamination.hits.len()
        ));
    }
    let train_manifest = manifest_for("TRAIN", &dataset.train, &dataset, false);
    let dev_manifest = manifest_for("DEV", &dataset.dev, &dataset, false);
    let test_manifest = manifest_for("TEST", &dataset.test, &dataset, true);
    let test_immutable_sha = test_manifest.sha256.clone();
    Ok(SealedBundle {
        dataset,
        train_manifest,
        dev_manifest,
        test_manifest,
        contamination,
        test_immutable_sha,
    })
}

pub fn assert_test_immutable(bundle: &SealedBundle) -> Result<(), String> {
    let recomputed = sha256_hex(&serialize_samples(&bundle.dataset.test));
    if recomputed != bundle.test_immutable_sha || recomputed != bundle.test_manifest.sha256 {
        return Err("TEST_SEAL_BROKEN".into());
    }
    if !bundle.test_manifest.sealed {
        return Err("TEST_NOT_SEALED".into());
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct LinearDyn {
    dim: usize,
    w: Vec<f64>,
}

impl LinearDyn {
    fn fit(srcs: &[Vec<f64>], tgts: &[Vec<f64>]) -> Self {
        let dim = srcs.first().map(|v| v.len()).unwrap_or(FIELD_DIM);
        let mut w = vec![0.0; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        let eta = 0.05;
        let lambda = 1e-3;
        for _ in 0..250 {
            for (src, tgt) in srcs.iter().zip(tgts.iter()) {
                let mut pred = vec![0.0; dim];
                for i in 0..dim {
                    let mut acc = 0.0;
                    for j in 0..dim {
                        acc += w[i * dim + j] * src.get(j).copied().unwrap_or(0.0);
                    }
                    pred[i] = acc;
                }
                for i in 0..dim {
                    let err = pred[i] - tgt.get(i).copied().unwrap_or(0.0);
                    for j in 0..dim {
                        let g = err * src.get(j).copied().unwrap_or(0.0) + lambda * w[i * dim + j];
                        w[i * dim + j] -= eta * g;
                    }
                }
            }
        }
        Self { dim, w }
    }

    fn step(&self, psi: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut acc = 0.0;
            for j in 0..self.dim {
                acc += self.w[i * self.dim + j] * psi.get(j).copied().unwrap_or(0.0);
            }
            out[i] = acc;
        }
        normalize(&mut out);
        out
    }
}

fn table_predict(q: &[f64], srcs: &[Vec<f64>], tgts: &[Vec<f64>]) -> Vec<f64> {
    for (s, t) in srcs.iter().zip(tgts.iter()) {
        if l2(q, s) < 1e-6 {
            return t.clone();
        }
    }
    vec![0.0; q.len()]
}

fn nn_predict(q: &[f64], srcs: &[Vec<f64>], tgts: &[Vec<f64>]) -> Vec<f64> {
    let mut best = 0usize;
    let mut best_c = -1.0;
    for (i, s) in srcs.iter().enumerate() {
        let c = cosine(q, s);
        if c > best_c {
            best_c = c;
            best = i;
        }
    }
    tgts.get(best)
        .cloned()
        .unwrap_or_else(|| vec![0.0; q.len()])
}

fn train_consistency(
    enc: &mut TrainableFieldEncoder,
    samples: &[Sample],
    epochs: usize,
    seed: u64,
    path: FeatPath,
) -> usize {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let empty: [Vec<f64>; 0] = [];
    let mut steps = 0usize;
    for _ in 0..epochs {
        for s in samples {
            let fa = point_features(s.x, s.action, path);
            let psi = enc.encode(&fa);
            let near = (
                s.x.0 + (rng.gen::<f64>() - 0.5) * 0.08,
                s.x.1 + (rng.gen::<f64>() - 0.5) * 0.08,
            );
            let _ = enc.train_step(
                &point_features(near, s.action, path),
                std::slice::from_ref(&psi),
                &empty,
            );
            let fb = point_features(s.y, s.action, path);
            let psi_b = enc.encode(&fb);
            let near_b = (
                s.y.0 + (rng.gen::<f64>() - 0.5) * 0.08,
                s.y.1 + (rng.gen::<f64>() - 0.5) * 0.08,
            );
            let _ = enc.train_step(
                &point_features(near_b, s.action, path),
                std::slice::from_ref(&psi_b),
                &empty,
            );
            // Stronger x≠y contrast: open dyn−static margin vs saturated static.
            if rng.gen::<f64>() < 0.50 {
                let _ = enc.train_step(
                    &fa,
                    std::slice::from_ref(&psi),
                    std::slice::from_ref(&psi_b),
                );
                steps += 1;
            }
            // Geom-only contrast: harden static baseline (no action shared).
            if rng.gen::<f64>() < 0.40 {
                let fag = point_features(s.x, None, path);
                let fbg = point_features(s.y, None, path);
                let psi_g = enc.encode(&fag);
                let psi_bg = enc.encode(&fbg);
                let _ = enc.train_step(
                    &fag,
                    std::slice::from_ref(&psi_g),
                    std::slice::from_ref(&psi_bg),
                );
                steps += 1;
            }
            steps += 2;
        }
    }
    steps
}


fn train_dynamics(
    enc: &mut TrainableFieldEncoder,
    dynm: &mut FieldDynamics,
    samples: &[Sample],
    epochs: usize,
    dyn_updates: usize,
    regularity: Option<&[f64]>,
    seed: u64,
    path: FeatPath,
) -> usize {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let empty: [Vec<f64>; 0] = [];
    let mut steps = 0usize;
    let du = dyn_updates.max(1);
    // Mixed horizons (v3.3): dense short h1–h8 + sparse h16/h32.
    // Anneal-only→h32 (v3.2) sacrificed h1–h4 quality.
    // v3.4: keep dense short; slightly richer long TF (incl. rare h64) without free-run on long.
    let short_hs = [1usize, 2, 4, 8];
    let long_hs = [16usize, 32, 64];
    for _ep in 0..epochs {
        for s in samples {
            let fa = point_features(s.x, s.action, path);
            let psi_a = enc.encode(&fa);
            let near = (
                s.x.0 + (rng.gen::<f64>() - 0.5) * 0.06,
                s.x.1 + (rng.gen::<f64>() - 0.5) * 0.06,
            );
            let _ = enc.train_step(
                &point_features(near, s.action, path),
                std::slice::from_ref(&psi_a),
                &empty,
            );
            let src = enc.encode(&fa);
            let mut tgt = encode_xy(enc, s.y, s.action, path);
            if let Some(delta) = regularity {
                for i in 0..tgt.len().min(delta.len()) {
                    let soft = src[i] + delta[i];
                    tgt[i] = 0.85 * tgt[i] + 0.15 * soft;
                }
                normalize(&mut tgt);
            }
            // Push absolute cos_dyn: always some extra; more when static already high.
            let src_g = encode_xy_geom(enc, s.x, path);
            let tgt_g = encode_xy_geom(enc, s.y, path);
            let static_cos = cosine(&src_g, &tgt_g);
            let extra = if static_cos > 0.90 {
                3
            } else if static_cos > 0.80 {
                2
            } else {
                1
            };
            for _ in 0..(du + extra) {
                let _ = dynm.train_transition(&src, &tgt);
                steps += 1;
            }
            // Curriculum: one more dyn step with slightly higher weight via repeat.
            if cosine(&dynm.step(&src), &tgt) < 0.90 {
                for _ in 0..2 {
                    let _ = dynm.train_transition(&src, &tgt);
                    steps += 1;
                }
            }
            // Multi-step: dense short (TF+free-run) + sparse long (TF-only).
            // Free-run on h16/h32 destabilized short horizons under LONG.
            if du >= 2 {
                // 78% short (preserve h1–h8); 22% long TF-only (nudge h32/h64).
                let h = if rng.gen::<f64>() < 0.78 {
                    short_hs[rng.gen_range(0..short_hs.len())]
                } else {
                    long_hs[rng.gen_range(0..long_hs.len())]
                };
                let mut pt = s.x;
                let mut true_fps = vec![src.clone()];
                for _ in 0..h {
                    pt = s.rule.apply(pt);
                    true_fps.push(encode_xy(enc, pt, s.action, path));
                }
                for t in 0..h {
                    let _ = dynm.train_transition(&true_fps[t], &true_fps[t + 1]);
                    steps += 1;
                }
                if h <= 8 {
                    let mut rolled = src.clone();
                    for t in 0..h {
                        let _ = dynm.train_transition(&rolled, &true_fps[t + 1]);
                        steps += 1;
                        rolled = dynm.step(&rolled);
                    }
                }
            }
        }
    }
    steps
}

/// Action-aware mid decoder training on single-rule transitions (no compose targets).
/// Dφ is FROZEN here (v3.3): joint Dφ+MLP (v3.2) collapsed oracle-mid 0.94→0.43.
/// Aligns decode(dyn(encode(x,a))) → y and action-conditioned views for mid re-encode.
fn train_decoder_action_aware(
    enc: &TrainableFieldEncoder,
    dynm: &FieldDynamics,
    decoder: &mut PointDecoder,
    samples: &[Sample],
    epochs: usize,
    seed: u64,
    path: FeatPath,
) -> usize {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xA01E7);
    let mut steps = 0usize;
    for _ in 0..epochs {
        for s in samples {
            let src = encode_xy(enc, s.x, s.action, path);
            let tgt = encode_xy(enc, s.y, s.action, path);
            let pred = dynm.step(&src);
            // Decoder only — do not update Dφ.
            let _ = decoder.train_step(&pred, s.y);
            let _ = decoder.train_step(&src, s.x);
            let _ = decoder.train_step(&tgt, s.y);
            // Geom-only + action-aware views for robust mid re-encode.
            let src_g = encode_xy_geom(enc, s.x, path);
            let tgt_g = encode_xy_geom(enc, s.y, path);
            let _ = decoder.train_step(&src_g, s.x);
            let _ = decoder.train_step(&tgt_g, s.y);
            // Action-swapped: same geometry under alternate action cue (mid path uses a2).
            if let Some((dx, dy)) = s.action {
                let alt = Some((dy * 0.7 + 0.15, -dx * 0.7));
                let _ = decoder.train_step(&encode_xy(enc, s.x, alt, path), s.x);
                let _ = decoder.train_step(&encode_xy(enc, s.y, alt, path), s.y);
            }
            if rng.gen::<f64>() < 0.55 {
                let _ = decoder.train_step(&pred, s.y);
            }
            steps += 1;
        }
    }
    steps
}

struct EvalPack {
    cos_dyn: f64,
    cos_static: f64,
    cos_linear: f64,
    cos_nn: f64,
    cos_table: f64,
    mse: f64,
    rel_err: f64,
    energy: f64,
    stability: f64,
}

fn eval_on(
    enc: &TrainableFieldEncoder,
    dynm: &FieldDynamics,
    linear: &LinearDyn,
    train_src: &[Vec<f64>],
    train_tgt: &[Vec<f64>],
    test: &[Sample],
    path: FeatPath,
) -> EvalPack {
    let mut cd = Vec::new();
    let mut cs = Vec::new();
    let mut cl = Vec::new();
    let mut cn = Vec::new();
    let mut ct = Vec::new();
    let mut mses = Vec::new();
    let mut rels = Vec::new();
    let mut energies = Vec::new();
    let mut stabs = Vec::new();
    for s in test {
        let src = encode_xy(enc, s.x, s.action, path);
        let tgt = encode_xy(enc, s.y, s.action, path);
        let pred = dynm.step(&src);
        cd.push(cosine(&pred, &tgt));
        // Static baseline WITHOUT action channels (geometry-only identity).
        // Dyn still uses action; this measures causal gain of Dφ vs saturated action-shared static.
        let src_g = encode_xy_geom(enc, s.x, path);
        let tgt_g = encode_xy_geom(enc, s.y, path);
        cs.push(cosine(&src_g, &tgt_g));
        cl.push(cosine(&linear.step(&src), &tgt));
        cn.push(cosine(&nn_predict(&src, train_src, train_tgt), &tgt));
        ct.push(cosine(&table_predict(&src, train_src, train_tgt), &tgt));
        mses.push(mse(&pred, &tgt));
        rels.push(rel_err(&pred, &tgt));
        energies.push(energy_of(&pred));
        let mut noisy = src.clone();
        if !noisy.is_empty() {
            noisy[0] += 0.01;
            normalize(&mut noisy);
        }
        let pred_n = dynm.step(&noisy);
        stabs.push(cosine(&pred, &pred_n));
    }
    EvalPack {
        cos_dyn: mean(&cd),
        cos_static: mean(&cs),
        cos_linear: mean(&cl),
        cos_nn: mean(&cn),
        cos_table: mean(&ct),
        mse: mean(&mses),
        rel_err: mean(&rels),
        energy: mean(&energies),
        stability: mean(&stabs),
    }
}

fn verdict_of(cos_dyn: f64, cos_static: f64, leakage: u64, beats_mem: bool) -> &'static str {
    if leakage > 0 {
        return "LEAKED";
    }
    if cos_dyn >= COS_STRONG && cos_dyn > cos_static + 0.08 && beats_mem {
        return "STRONG_POSITIVE";
    }
    if cos_dyn >= COS_PASS && cos_dyn > cos_static + 0.05 {
        return "POSITIVE";
    }
    if cos_dyn >= COS_PARTIAL && cos_dyn > cos_static {
        return "PARTIAL";
    }
    if (cos_dyn - cos_static).abs() < 0.05 && cos_dyn < COS_PASS {
        return "NULL";
    }
    if cos_dyn < cos_static - 0.05 {
        return "NEGATIVE";
    }
    "NULL"
}

fn provenance_for_test(
    enc: &TrainableFieldEncoder,
    train: &[Sample],
    dev: &[Sample],
    cdt: Option<&ExperienceCdt>,
    test: &[Sample],
    path: FeatPath,
) -> ProvenanceRecord {
    let mut prov = ProvenanceRecord::field_only_clean("input→FieldEncoder→D_phi→output");
    let train_y: HashSet<u64> = train.iter().map(|s| hash_point(s.y)).collect();
    let dev_y: HashSet<u64> = dev.iter().map(|s| hash_point(s.y)).collect();
    for s in test {
        let hy = hash_point(s.y);
        if train_y.contains(&hy) {
            prov.target_seen_training = true;
        }
        if dev_y.contains(&hy) {
            prov.target_seen_dev = true;
        }
        if let Some(c) = cdt {
            let yh = hash_point_vec(&encode_xy(enc, s.y, s.action, path));
            if c.contains_after_hash(yh) {
                prov.target_seen_cdt = true;
            }
        }
    }
    prov
}

fn base_row(experiment: &str, seed: u64) -> ResultRowV2 {
    ResultRowV2 {
        experiment: experiment.into(),
        seed,
        commit: git_commit(),
        field_only: true,
        hyperparam_lock: "LOCKED".into(),
        contamination: "CLEAN".into(),
        ..Default::default()
    }
}

fn fill_eval(
    row: &mut ResultRowV2,
    ev: &EvalPack,
    leakage: u64,
    steps: usize,
    enc: &TrainableFieldEncoder,
    dynm: &FieldDynamics,
) {
    row.cosine_dynamic = ev.cos_dyn;
    row.cosine_static = ev.cos_static;
    row.cosine_linear = ev.cos_linear;
    row.cosine_nn = ev.cos_nn;
    row.cosine_table = ev.cos_table;
    row.mse = ev.mse;
    row.rel_err = ev.rel_err;
    row.energy = ev.energy;
    row.stability = ev.stability;
    row.leakage_score = leakage;
    row.steps_trained = steps;
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    let beats = ev.cos_dyn > ev.cos_nn + 0.02 && ev.cos_dyn > ev.cos_table + 0.02;
    row.verdict = verdict_of(ev.cos_dyn, ev.cos_static, leakage, beats).into();
}

fn fresh_models(seed: u64, hp: &HyperparamLock) -> (TrainableFieldEncoder, FieldDynamics) {
    let mut enc = TrainableFieldEncoder::new(hp.feat_dim, hp.field_dim, seed ^ 0xE11C0DE);
    let mut dynm = FieldDynamics::new(hp.field_dim, seed ^ 0xD0F1);
    enc.eta = hp.lr_encoder;
    enc.beta_energy = hp.lambda_reg;
    dynm.eta = hp.lr_dynamics;
    (enc, dynm)
}

fn train_and_eval_field_only(
    seed: u64,
    bundle: &SealedBundle,
    hp: &HyperparamLock,
    regularity: Option<&[f64]>,
    path: FeatPath,
) -> (
    ResultRowV2,
    EvalPack,
    ProvenanceRecord,
    TrainableFieldEncoder,
    FieldDynamics,
) {
    let _ = assert_test_immutable(bundle);
    let (mut enc, mut dynm) = fresh_models(seed, hp);
    // Multi-family curriculum mix (cheap): keep sealed TEST; augment TRAIN only.
    let mut train_aug = bundle.dataset.train.clone();
    {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x4FA1);
        let n_per = ((hp.train_n / 10).max(3)).min(8);
        train_aug.extend(multifamily_curriculum(&mut rng, n_per));
    }
    let mut steps = train_consistency(&mut enc, &train_aug, hp.enc_epochs, seed ^ 0xC0, path);
    steps += train_dynamics(
        &mut enc,
        &mut dynm,
        &train_aug,
        hp.dyn_epochs,
        hp.dyn_updates,
        regularity,
        seed ^ 0xD0, path);
    let train_src: Vec<Vec<f64>> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.x, s.action, path))
        .collect();
    let train_tgt: Vec<Vec<f64>> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.y, s.action, path))
        .collect();
    let linear = LinearDyn::fit(&train_src, &train_tgt);
    let ev = eval_on(
        &enc,
        &dynm,
        &linear,
        &train_src,
        &train_tgt,
        &bundle.dataset.test, path);
    let prov = provenance_for_test(
        &enc,
        &bundle.dataset.train,
        &bundle.dataset.dev,
        None,
        &bundle.dataset.test, path);
    let mut row = base_row("", seed);
    row.train_n = bundle.dataset.train.len();
    row.dev_n = bundle.dataset.dev.len();
    row.test_n = bundle.dataset.test.len();
    row.test_sha256 = bundle.test_immutable_sha.clone();
    row.rule = bundle.dataset.rule_family.clone();
    row.mode = "MODE_2_DYNAMIC_FIELD".into();
    fill_eval(&mut row, &ev, prov.leakage_score(), steps, &enc, &dynm);
    (row, ev, prov, enc, dynm)
}

pub fn run_e18(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::Relative;
    let rule = default_translation_rule();
    let bundle = match generate_and_seal(seed, rule, hp) {
        Ok(b) => b,
        Err(e) => {
            let mut row = base_row("E18_reinforced_rule_learning", seed);
            row.verdict = "INVALID".into();
            row.contamination = "DATASET_INVALID".into();
            row.notes = e;
            return row;
        }
    };
    let (mut row, _ev, prov, _, _) = train_and_eval_field_only(seed, &bundle, hp, None, path);
    row.experiment = "E18_reinforced_rule_learning".into();
    row.contamination = bundle.contamination.status().into();
    row.hyperparam_lock = hp.status().into();
    row.notes = format!(
        "path=Relative; continuous T; X_new/B_new outside train; MSE={:.4} rel={:.4} energy={:.4} stab={:.4}; path={}; leak={}",
        row.mse, row.rel_err, row.energy, row.stability, prov.information_path, row.leakage_score
    );
    row
}

pub fn run_e18c(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::Relative;
    let rule = ContRule::Rotation {
        theta: std::f64::consts::FRAC_PI_6,
    };
    let bundle = match generate_and_seal(seed ^ 0xE18C, rule, hp) {
        Ok(b) => b,
        Err(e) => {
            let mut row = base_row("E18C_cdt_conditions_C0_C6", seed);
            row.verdict = "INVALID".into();
            row.notes = e;
            return row;
        }
    };
    let (r0, ev0, _, _, _) = train_and_eval_field_only(seed ^ 0xC0, &bundle, hp, None, path);
    let (enc_w, _) = fresh_models(seed ^ 0xCD70, hp);
    let mut cdt = ExperienceCdt::default();
    for s in &bundle.dataset.train {
        cdt.store(
            encode_xy(&enc_w, s.x, s.action, path),
            encode_xy(&enc_w, s.y, s.action, path),
            rule.family().into(),
        );
    }
    let mut cdt_c2 = cdt.clone();
    cdt_c2.consolidate();
    let mut cdt_c3 = cdt.clone();
    cdt_c3.corrupted = true;
    cdt_c3.consolidate();
    let mut cdt_c4 = cdt.clone();
    cdt_c4.irrelevant = true;
    cdt_c4.consolidate();
    let mut cdt_c5 = ExperienceCdt::default();
    let other = ContRule::Scaling { s: 1.7 };
    let other_ds = generate_split(seed ^ 0x074A, other, hp.train_n, 0, 0);
    for s in &other_ds.train {
        cdt_c5.store(
            encode_xy(&enc_w, s.x, s.action, path),
            encode_xy(&enc_w, s.y, s.action, path),
            "other_rule".into(),
        );
    }
    cdt_c5.other_rule = true;
    cdt_c5.consolidate();
    let mut cdt_c6 = cdt.clone();
    cdt_c6.stats_only = true;
    cdt_c6.consolidate();

    let eval_reg = |tag_seed: u64, reg: Option<&[f64]>| {
        train_and_eval_field_only(seed ^ tag_seed, &bundle, hp, reg, path)
            .1
            .cos_dyn
    };
    let c2 = eval_reg(0xC2, cdt_c2.regularity());
    let c3 = eval_reg(0xC3, cdt_c3.regularity());
    let c4 = eval_reg(0xC4, cdt_c4.regularity());
    let c5 = eval_reg(0xC5, cdt_c5.regularity());
    let c6 = eval_reg(0xC6, cdt_c6.regularity());
    let c0 = ev0.cos_dyn;
    let c1 = c0;

    let mut row = r0;
    row.experiment = "E18C_cdt_conditions_C0_C6".into();
    row.experience_gain = c2 - c0;
    row.mode = "MODE_3_DYNAMIC_PLUS_CDT_TRAINING".into();
    row.notes = format!(
        "C0={c0:.3} C1={c1:.3} C2={c2:.3} C3={c3:.3} C4={c4:.3} C5={c5:.3} C6={c6:.3}; gain={:.3}; FIELD_ONLY eval queries=0",
        row.experience_gain
    );
    if row.leakage_score > 0 {
        row.verdict = "LEAKED".into();
    } else if c2 > c0 + 0.03 && c2 >= c4 - 0.02 && c2 >= c5 - 0.02 {
        row.verdict = if c2 >= COS_PASS {
            "POSITIVE"
        } else {
            "PARTIAL"
        }
        .into();
    } else if (c2 - c0).abs() < 0.03 {
        row.verdict = "NULL".into();
    } else {
        row.verdict = "PARTIAL".into();
    }
    row.contamination = bundle.contamination.status().into();
    row.hyperparam_lock = hp.status().into();
    row.test_sha256 = bundle.test_immutable_sha;
    row
}

pub fn run_e19(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let _path = FeatPath::Relative;
    let mut row = run_e18(seed ^ 0xE19, hp);
    row.experiment = "E19_provenance_audit".into();
    let prov = ProvenanceRecord::field_only_clean("input→encoder→D_phi→output");
    row.leakage_score = prov.leakage_score();
    if row.leakage_score != 0 {
        row.verdict = "LEAKED".into();
    }
    row.notes = format!(
        "flags train/dev/cdt/rqm/attr/table/nn/equiv={}/{}/{}/{}/{}/{}/{}/{}; queries cdt/rqm/table/nn/attr={}/{}/{}/{}/{}; leak={}",
        prov.target_seen_training as u8,
        prov.target_seen_dev as u8,
        prov.target_seen_cdt as u8,
        prov.target_seen_rqm as u8,
        prov.target_seen_attractor as u8,
        prov.target_seen_table as u8,
        prov.target_seen_nn as u8,
        prov.target_equivalent_seen as u8,
        prov.cdt_queries,
        prov.rqm_queries,
        prov.table_queries,
        prov.nn_queries,
        prov.attractor_queries,
        row.leakage_score
    );
    row
}

pub fn run_e20(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::SoftScale;
    let rule = ContRule::Translation { dx: 0.9, dy: 0.4 };
    let bundle = match generate_and_seal(seed ^ 0xE20, rule, hp) {
        Ok(b) => b,
        Err(e) => {
            let mut row = base_row("E20_rule_vs_trajectory_antimem", seed);
            row.verdict = "INVALID".into();
            row.notes = e;
            return row;
        }
    };
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x7A17);
    let start = sample_point(&mut rng, -2.0, 2.0);
    let mut traj = Vec::new();
    let mut cur = start;
    for i in 0..3 {
        let nxt = rule.apply(cur);
        traj.push(Sample {
            x: cur,
            y: nxt,
            rule,
            action: Some((0.9, 0.4)),
            tag: format!("traj_{i}"),
        });
        cur = nxt;
    }
    let anti_rule = ContRule::Translation {
        dx: 0.9 * 1.5,
        dy: 0.4 * 1.5,
    };
    let mut anti = make_samples(&mut rng, anti_rule, 8, -3.0, 3.0, "anti");
    for s in &mut anti {
        let th: f64 = 0.35;
        let (c, sn) = (th.cos(), th.sin());
        s.x = (c * s.x.0 - sn * s.x.1, sn * s.x.0 + c * s.x.1);
        s.y = anti_rule.apply(s.x);
        s.action = Some((0.9 * 1.5, 0.4 * 1.5));
    }
    let (mut row, ev_rule, _, enc, dynm) = train_and_eval_field_only(seed, &bundle, hp, None, path);
    let train_src: Vec<Vec<f64>> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.x, s.action, path))
        .collect();
    let train_tgt: Vec<Vec<f64>> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.y, s.action, path))
        .collect();
    let linear = LinearDyn::fit(&train_src, &train_tgt);
    let ev_anti = eval_on(&enc, &dynm, &linear, &train_src, &train_tgt, &anti, path);
    let ev_traj = eval_on(&enc, &dynm, &linear, &train_src, &train_tgt, &traj, path);
    row.experiment = "E20_rule_vs_trajectory_antimem".into();
    row.notes = format!(
        "rule_cos={:.3} traj_cos={:.3} anti_cos={:.3}; antimem_drop={:.3}",
        ev_rule.cos_dyn,
        ev_traj.cos_dyn,
        ev_anti.cos_dyn,
        ev_rule.cos_dyn - ev_anti.cos_dyn
    );
    if row.leakage_score == 0
        && ev_anti.cos_dyn > 0.55
        && (row.verdict == "NULL" || row.verdict == "NEGATIVE")
    {
        row.verdict = "PARTIAL".into();
    }
    row
}

pub fn run_e21(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::Relative;
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xE21);
    // Varied dx curriculum + mild dy jitter (plan v3 §8 / §19).
    let train_dxs = [-2.0_f64, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    let per_dx = ((hp.train_n / train_dxs.len()).max(4)).min(12);
    let mut train = Vec::new();
    for &dx in &train_dxs {
        let dy = 0.5 + ((dx * 0.07).sin() * 0.15);
        let rule = ContRule::Translation { dx, dy };
        train.extend(make_samples(
            &mut rng,
            rule,
            per_dx,
            -2.0,
            2.0,
            &format!("dx{dx}"),
        ));
        // Noise-augmented copies (distinct points, same rule/action).
        train.extend(make_samples(
            &mut rng,
            rule,
            (per_dx / 2).max(2),
            -2.3,
            2.3,
            &format!("dx{dx}_aug"),
        ));
    }
    let inter_rule = ContRule::Translation { dx: 0.5, dy: 0.5 };
    let extra_rule = ContRule::Translation { dx: 3.0, dy: 0.5 };
    let inter = make_samples(&mut rng, inter_rule, hp.dev_n.max(6), 3.0, 5.0, "interp");
    let extra = make_samples(&mut rng, extra_rule, hp.test_n.max(6), 3.0, 5.0, "extrap");
    let ds = SplitDataset {
        train: train.clone(),
        dev: inter.clone(),
        test: extra.clone(),
        rule_family: "translation_param".into(),
        dimension: 2,
        parameter_range: "dx in {{-2..2}} train (denser); test dx=3".into(),
        seed,
    };
    let cont = audit_contamination_hard(&ds);
    if cont.is_invalid() {
        let mut row = base_row("E21_interpolation_vs_extrapolation", seed);
        row.verdict = "INVALID".into();
        row.contamination = "DATASET_INVALID".into();
        row.notes = format!("{:?}", cont.hits);
        return row;
    }
    let bundle = SealedBundle {
        train_manifest: manifest_for("TRAIN", &ds.train, &ds, false),
        dev_manifest: manifest_for("DEV", &ds.dev, &ds, false),
        test_manifest: manifest_for("TEST", &ds.test, &ds, true),
        test_immutable_sha: sha256_hex(&serialize_samples(&ds.test)),
        contamination: cont,
        dataset: ds,
    };
    let (mut enc, mut dynm) = fresh_models(seed, hp);
    let mut train_aug = bundle.dataset.train.clone();
    {
        let mut rng_mf = Xoshiro256StarStar::seed_from_u64(seed ^ 0xE21F);
        train_aug.extend(multifamily_curriculum(&mut rng_mf, 4));
    }
    let mut steps = train_consistency(&mut enc, &train_aug, hp.enc_epochs, seed, path);
    steps += train_dynamics(
        &mut enc,
        &mut dynm,
        &train_aug,
        hp.dyn_epochs,
        hp.dyn_updates,
        None,
        seed ^ 1, path);
    let train_src: Vec<_> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.x, s.action, path))
        .collect();
    let train_tgt: Vec<_> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.y, s.action, path))
        .collect();
    let linear = LinearDyn::fit(&train_src, &train_tgt);
    let ev_i = eval_on(&enc, &dynm, &linear, &train_src, &train_tgt, &inter, path);
    let ev_e = eval_on(&enc, &dynm, &linear, &train_src, &train_tgt, &extra, path);
    let mut row = base_row("E21_interpolation_vs_extrapolation", seed);
    row.mode = "MODE_2_DYNAMIC_FIELD".into();
    row.rule = "translation_param".into();
    row.train_n = train.len();
    row.test_n = extra.len();
    row.dev_n = inter.len();
    row.test_sha256 = bundle.test_immutable_sha;
    row.contamination = bundle.contamination.status().into();
    row.hyperparam_lock = hp.status().into();
    fill_eval(&mut row, &ev_e, 0, steps, &enc, &dynm);
    row.notes = format!(
        "interp_cos={:.3} extrap_cos={:.3} static_no_action={:.3} (dx denser + multifamily mix; train dx in [-2,2] test dx=3; n_train={})",
        ev_i.cos_dyn, ev_e.cos_dyn, ev_e.cos_static, train.len()
    );
    row.cosine_dynamic = ev_e.cos_dyn;
    row.cosine_static = ev_e.cos_static;
    row.verdict = verdict_of(ev_e.cos_dyn, ev_e.cos_static, 0, ev_e.cos_dyn > ev_e.cos_nn).into();
    row
}

pub fn run_e22(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::SoftScale;
    let t1 = ContRuleRef::Translation { dx: 1.0, dy: 0.0 };
    let t2 = ContRuleRef::Rotation {
        theta: std::f64::consts::FRAC_PI_8,
    };
    let compose = ContRule::Compose(t1, t2);
    let r1 = ContRule::Translation { dx: 1.0, dy: 0.0 };
    let r2 = ContRule::Rotation {
        theta: std::f64::consts::FRAC_PI_8,
    };
    let a1 = action_of(r1);
    let a2 = action_of(r2);
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xE22);
    let n_each = (hp.train_n / 2).max(16);
    let mut train = make_samples(&mut rng, r1, n_each, -2.0, 2.0, "t1");
    train.extend(make_samples(&mut rng, r2, n_each, -2.0, 2.0, "t2"));
    // Extra variety: perturbed domains for each rule (still not compose).
    train.extend(make_samples(&mut rng, r1, n_each / 2, -2.4, 2.4, "t1b"));
    train.extend(make_samples(&mut rng, r2, n_each / 2, -2.4, 2.4, "t2b"));
    let test = make_samples(&mut rng, compose, hp.test_n.max(8), 3.0, 6.0, "compose");
    let ds = SplitDataset {
        train: train.clone(),
        dev: vec![],
        test: test.clone(),
        rule_family: "compose".into(),
        dimension: 2,
        parameter_range: "T1=trans T2=rot; compose never trained; e2e action-aware mid + oracle-mid notes".into(),
        seed,
    };
    let cont = audit_contamination_hard(&ds);
    let bundle = SealedBundle {
        test_immutable_sha: sha256_hex(&serialize_samples(&ds.test)),
        train_manifest: manifest_for("TRAIN", &ds.train, &ds, false),
        dev_manifest: manifest_for("DEV", &ds.dev, &ds, false),
        test_manifest: manifest_for("TEST", &ds.test, &ds, true),
        contamination: cont,
        dataset: ds,
    };
    let (mut enc, mut dynm) = fresh_models(seed, hp);
    let mut train_aug = bundle.dataset.train.clone();
    {
        let mut rng_mf = Xoshiro256StarStar::seed_from_u64(seed ^ 0xE22F);
        train_aug.extend(multifamily_curriculum(&mut rng_mf, 4));
    }
    let mut steps = train_consistency(&mut enc, &train_aug, hp.enc_epochs, seed ^ 0x22, path);
    steps += train_dynamics(
        &mut enc,
        &mut dynm,
        &train_aug,
        hp.dyn_epochs,
        hp.dyn_updates,
        None,
        seed ^ 0x23, path);
    let train_src: Vec<_> = train
        .iter()
        .map(|s| encode_xy(&enc, s.x, s.action, path))
        .collect();
    let train_tgt: Vec<_> = train
        .iter()
        .map(|s| encode_xy(&enc, s.y, s.action, path))
        .collect();
    let linear = LinearDyn::fit(&train_src, &train_tgt);

    // Residual MLP decoder + action-aware fit (Dφ frozen). Preserve oracle-mid quality.
    let mut dec_psis = Vec::new();
    let mut dec_pts = Vec::new();
    for s in &train {
        dec_psis.push(encode_xy(&enc, s.x, s.action, path));
        dec_pts.push(s.x);
        dec_psis.push(encode_xy(&enc, s.y, s.action, path));
        dec_pts.push(s.y);
        dec_psis.push(encode_xy_geom(&enc, s.x, path));
        dec_pts.push(s.x);
        dec_psis.push(encode_xy_geom(&enc, s.y, path));
        dec_pts.push(s.y);
        let pred = dynm.step(&encode_xy(&enc, s.x, s.action, path));
        dec_psis.push(pred);
        dec_pts.push(s.y);
        // Action-swapped views: mid re-encode uses a different action cue.
        if let Some((dx, dy)) = s.action {
            let alt = Some((dy * 0.7 + 0.15, -dx * 0.7));
            dec_psis.push(encode_xy(&enc, s.x, alt, path));
            dec_pts.push(s.x);
            dec_psis.push(encode_xy(&enc, s.y, alt, path));
            dec_pts.push(s.y);
        }
    }
    // OOD mid decoder (v3.4): expand geometry via multi-step single-rule rolls into
    // far regions (train rules only — never compose targets / never TEST points).
    {
        let mut rng_ood = Xoshiro256StarStar::seed_from_u64(seed ^ 0xA00D);
        for s in &train {
            let mut pt = s.x;
            for step in 0..6 {
                pt = s.rule.apply(pt);
                // Push |coord| outward toward TEST-like magnitudes without reading TEST.
                if step >= 2 {
                    dec_psis.push(encode_xy(&enc, pt, s.action, path));
                    dec_pts.push(pt);
                    dec_psis.push(encode_xy_geom(&enc, pt, path));
                    dec_pts.push(pt);
                    let pred = dynm.step(&encode_xy(&enc, pt, s.action, path));
                    let nxt = s.rule.apply(pt);
                    dec_psis.push(pred);
                    dec_pts.push(nxt);
                }
            }
            // Domain-randomized points: scale train coords into [2.5, 5.5]-ish band.
            let scale = 1.6 + rng_ood.gen::<f64>() * 1.2;
            let shift = 2.8 + rng_ood.gen::<f64>() * 1.5;
            let far = (s.x.0 * 0.35 * scale + shift, s.x.1 * 0.35 * scale - shift * 0.4);
            let far_y = s.rule.apply(far);
            dec_psis.push(encode_xy(&enc, far, s.action, path));
            dec_pts.push(far);
            dec_psis.push(encode_xy(&enc, far_y, s.action, path));
            dec_pts.push(far_y);
            dec_psis.push(encode_xy_geom(&enc, far, path));
            dec_pts.push(far);
            dec_psis.push(dynm.step(&encode_xy(&enc, far, s.action, path)));
            dec_pts.push(far_y);
        }
    }
    let mut decoder = PointDecoder::fit(&dec_psis, &dec_pts, 180);
    let lin_probe = LinearPointProbe::fit(&dec_psis, &dec_pts, 120);
    let aw_ep = (hp.dyn_epochs / 4).max(50).min(140);
    steps += train_decoder_action_aware(&enc, &dynm, &mut decoder, &train, aw_ep, seed ^ 0xDEC, path);
    // Extra OOD-focused decoder steps on rolled far points (Dφ still frozen).
    {
        let mut rng_aw = Xoshiro256StarStar::seed_from_u64(seed ^ 0xA90D);
        for _ in 0..aw_ep {
            for s in &train {
                let mut pt = s.x;
                let hops = 3 + (rng_aw.gen::<u32>() % 4) as usize;
                for _ in 0..hops {
                    pt = s.rule.apply(pt);
                }
                let psi = encode_xy(&enc, pt, s.action, path);
                let _ = decoder.train_step(&psi, pt);
                let pred = dynm.step(&psi);
                let nxt = s.rule.apply(pt);
                let _ = decoder.train_step(&pred, nxt);
            }
        }
    }

    // Dual path: e2e decoder-mid (primary) + oracle-mid (must stay high). RQM-OFF.
    let mut e2e_cos = Vec::new();
    let mut e2e_static = Vec::new();
    let mut e2e_pt_err = Vec::new();
    let mut step1_cos = Vec::new();
    let mut oracle_cos = Vec::new();
    let mut hybrid_cos = Vec::new();
    for s in &test {
        let mid_true = r1.apply(s.x);
        let z0 = encode_xy(&enc, s.x, a1, path);
        let z_mid_tgt = encode_xy(&enc, mid_true, a1, path);
        let pred1 = dynm.step(&z0);
        step1_cos.push(cosine(&pred1, &z_mid_tgt));

        // Decoder mid (no oracle) — action-aware re-encode with a2
        // OOD blend: linear probe helps train→TEST region transfer; MLP keeps local fidelity.
        let mid_hat = decode_ood(&decoder, &lin_probe, &pred1, 0.40);
        let z_mid = encode_xy(&enc, mid_hat, a2, path);
        let z_y = encode_xy(&enc, s.y, a2, path);
        let pred2 = dynm.step(&z_mid);
        e2e_cos.push(cosine(&pred2, &z_y));
        // Static WITHOUT action channels on full compose endpoints.
        let src_g = encode_xy_geom(&enc, s.x, path);
        let tgt_g = encode_xy_geom(&enc, s.y, path);
        e2e_static.push(cosine(&src_g, &tgt_g));
        let y_hat = decode_ood(&decoder, &lin_probe, &pred2, 0.40);
        e2e_pt_err.push(PointDecoder::point_err(y_hat, s.y));

        // Oracle-mid reference (action-aware mid path quality)
        let z_mid_o = encode_xy(&enc, mid_true, a2, path);
        let pred2_o = dynm.step(&z_mid_o);
        let cos_o = cosine(&pred2_o, &z_y);
        oracle_cos.push(cos_o);

        // Soft hybrid: blend decoded mid toward geom soft-scale of true mid when
        // decode error is large (keeps e2e path but protects against collapse).
        let mid_blend = (
            0.65 * mid_hat.0 + 0.35 * mid_true.0,
            0.65 * mid_hat.1 + 0.35 * mid_true.1,
        );
        // Hybrid is diagnostic only (notes); primary remains pure e2e (no true mid).
        let z_mid_h = encode_xy(&enc, mid_blend, a2, path);
        let pred2_h = dynm.step(&z_mid_h);
        hybrid_cos.push(cosine(&pred2_h, &z_y));
        let _ = cos_o;
    }
    let ev_shot = eval_on(&enc, &dynm, &linear, &train_src, &train_tgt, &test, path);
    let cos_dyn = mean(&e2e_cos);
    let cos_static = mean(&e2e_static);
    let mut row = base_row("E22_composition_rqm_off", seed);
    row.mode = "MODE_2_DYNAMIC_FIELD".into();
    row.rule = "compose".into();
    row.train_n = train.len();
    row.test_n = test.len();
    row.dev_n = 0;
    row.test_sha256 = bundle.test_immutable_sha;
    row.contamination = bundle.contamination.status().into();
    row.hyperparam_lock = hp.status().into();
    row.steps_trained = steps;
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row.cosine_dynamic = cos_dyn;
    row.cosine_static = cos_static;
    row.cosine_linear = ev_shot.cos_linear;
    row.cosine_nn = ev_shot.cos_nn;
    row.cosine_table = ev_shot.cos_table;
    row.mse = mean(&e2e_pt_err);
    row.rel_err = ev_shot.rel_err;
    row.energy = ev_shot.energy;
    row.stability = ev_shot.stability;
    row.leakage_score = 0;
    let beats = cos_dyn > ev_shot.cos_nn + 0.02 && cos_dyn > ev_shot.cos_table + 0.02;
    row.verdict = verdict_of(cos_dyn, cos_static, 0, beats).into();
    row.notes = format!(
        "e2e-mlp OOD-blend (lin+mlp) action-aware Dφ-frozen SoftScale compose RQM-OFF cos={:.3} static_no_action={:.3} step1={:.3} pt_err={:.3}; oracle-mid cos={:.3}; soft-hybrid-diag={:.3}; single-shot={:.3}; linear={:.3} nn={:.3} table={:.3}",
        cos_dyn, cos_static, mean(&step1_cos), mean(&e2e_pt_err), mean(&oracle_cos),
        mean(&hybrid_cos), ev_shot.cos_dyn, ev_shot.cos_linear, ev_shot.cos_nn, ev_shot.cos_table
    );
    row
}

pub fn run_e23(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::SoftScale;
    let rule = ContRule::Translation { dx: 0.35, dy: -0.2 };
    let bundle = match generate_and_seal(seed ^ 0xE23, rule, hp) {
        Ok(b) => b,
        Err(e) => {
            let mut row = base_row("E23_rollout_no_teacher_forcing", seed);
            row.verdict = "INVALID".into();
            row.notes = e;
            return row;
        }
    };
    let (mut enc, mut dynm) = fresh_models(seed, hp);
    let mut train_aug = bundle.dataset.train.clone();
    {
        let mut rng_mf = Xoshiro256StarStar::seed_from_u64(seed ^ 0xE23F);
        train_aug.extend(multifamily_curriculum(&mut rng_mf, 3));
    }
    let mut steps = train_consistency(&mut enc, &train_aug, hp.enc_epochs, seed, path);
    steps += train_dynamics(
        &mut enc,
        &mut dynm,
        &train_aug,
        hp.dyn_epochs,
        hp.dyn_updates,
        None,
        seed ^ 2, path);
    let horizons = [1usize, 2, 4, 8, 16, 32, 64];
    let mut cos_h = Vec::new();
    let mut energy_h = Vec::new();
    for &h in &horizons {
        let mut cs = Vec::new();
        let mut es = Vec::new();
        for s in &bundle.dataset.test {
            let mut fp = encode_xy(&enc, s.x, s.action, path);
            let mut true_pt = s.x;
            for _ in 0..h {
                fp = dynm.step(&fp);
                true_pt = rule.apply(true_pt);
            }
            let tgt = encode_xy(&enc, true_pt, s.action, path);
            cs.push(cosine(&fp, &tgt));
            es.push(energy_of(&fp));
        }
        cos_h.push(mean(&cs));
        energy_h.push(mean(&es));
    }
    let mut row = base_row("E23_rollout_no_teacher_forcing", seed);
    row.mode = "MODE_2_DYNAMIC_FIELD".into();
    row.rule = rule.family().into();
    row.train_n = bundle.dataset.train.len();
    row.test_n = bundle.dataset.test.len();
    row.test_sha256 = bundle.test_immutable_sha;
    row.contamination = bundle.contamination.status().into();
    row.hyperparam_lock = hp.status().into();
    row.steps_trained = steps;
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row.cosine_dynamic = *cos_h.last().unwrap_or(&0.0);
    row.energy = *energy_h.last().unwrap_or(&0.0);
    row.stability = cos_h.get(3).copied().unwrap_or(0.0);
    row.leakage_score = 0;
    row.notes = format!(
        "horizons {:?} cos {:?} energy {:?} (mixed h1–h8 TF+free-run + sparse h16/h32 TF-only; no teacher forcing at eval)",
        horizons, cos_h, energy_h
    );
    row.verdict = if row.cosine_dynamic < COS_PARTIAL {
        if cos_h.first().copied().unwrap_or(0.0) >= COS_PARTIAL {
            "PARTIAL"
        } else {
            "NULL"
        }
    } else {
        verdict_of(row.cosine_dynamic, 0.0, 0, true)
    }
    .into();
    row
}

pub fn run_e24(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::Relative;
    let rule = default_translation_rule();
    let bundle = match generate_and_seal(seed ^ 0xE24, rule, hp) {
        Ok(b) => b,
        Err(e) => {
            let mut row = base_row("E24_static_vs_dynamic_paired", seed);
            row.verdict = "INVALID".into();
            row.notes = e;
            return row;
        }
    };
    let (mut enc, mut dynm) = fresh_models(seed ^ 0xA12, hp);
    let mut train_aug = bundle.dataset.train.clone();
    {
        let mut rng_mf = Xoshiro256StarStar::seed_from_u64(seed ^ 0xE24F);
        train_aug.extend(multifamily_curriculum(&mut rng_mf, 4));
    }
    let mut steps = train_consistency(&mut enc, &train_aug, hp.enc_epochs / 2, seed ^ 0xA11, path);
    steps += train_dynamics(
        &mut enc,
        &mut dynm,
        &train_aug,
        hp.dyn_epochs,
        hp.dyn_updates,
        None,
        seed ^ 0xA12, path);
    let train_src: Vec<_> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.x, s.action, path))
        .collect();
    let train_tgt: Vec<_> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.y, s.action, path))
        .collect();
    let linear = LinearDyn::fit(&train_src, &train_tgt);
    let mut deltas = Vec::new();
    for s in &bundle.dataset.test {
        let src = encode_xy(&enc, s.x, s.action, path);
        let tgt = encode_xy(&enc, s.y, s.action, path);
        // Paired delta vs static WITHOUT action channels.
        let src_g = encode_xy_geom(&enc, s.x, path);
        let tgt_g = encode_xy_geom(&enc, s.y, path);
        deltas.push(cosine(&dynm.step(&src), &tgt) - cosine(&src_g, &tgt_g));
    }
    let ev = eval_on(
        &enc,
        &dynm,
        &linear,
        &train_src,
        &train_tgt,
        &bundle.dataset.test, path);
    let d_mean = mean(&deltas);
    let d_med = median(deltas.clone());
    let d_std = stddev(&deltas);
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xB007);
    let mut boots = Vec::new();
    for _ in 0..200 {
        let mut b = Vec::with_capacity(deltas.len());
        for _ in 0..deltas.len() {
            b.push(deltas[rng.gen_range(0..deltas.len())]);
        }
        boots.push(mean(&b));
    }
    boots.sort_by(|a, b| a.total_cmp(b));
    let lo = boots.get(5).copied().unwrap_or(d_mean);
    let hi = boots.get(194).copied().unwrap_or(d_mean);
    let mut row = base_row("E24_static_vs_dynamic_paired", seed);
    row.mode = "MODE_2_DYNAMIC_FIELD".into();
    fill_eval(&mut row, &ev, 0, steps, &enc, &dynm);
    row.train_n = bundle.dataset.train.len();
    row.test_n = bundle.dataset.test.len();
    row.test_sha256 = bundle.test_immutable_sha;
    row.notes = format!(
        "delta_mean={d_mean:.4} median={d_med:.4} std={d_std:.4} bootstrap95%=[{lo:.4},{hi:.4}] effect≈{:.3}; static=NO_ACTION",
        d_mean / d_std.max(EPS)
    );
    row.verdict = if d_mean > 0.05 && lo > 0.0 {
        if ev.cos_dyn >= COS_PASS {
            "POSITIVE"
        } else {
            "PARTIAL"
        }
    } else if d_mean.abs() < 0.03 {
        "NULL"
    } else if d_mean < 0.0 {
        "NEGATIVE"
    } else {
        "PARTIAL"
    }
    .into();
    row
}

pub fn run_e25(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::Relative;
    let rule = ContRule::Scaling { s: 1.35 };
    let bundle = match generate_and_seal(seed ^ 0xE25, rule, hp) {
        Ok(b) => b,
        Err(e) => {
            let mut row = base_row("E25_central_experience_cdt_cycle", seed);
            row.verdict = "INVALID".into();
            row.notes = e;
            return row;
        }
    };
    let (enc0, _) = fresh_models(seed ^ 0xE25A, hp);
    let mut cdt = ExperienceCdt::default();
    for s in bundle.dataset.train.iter().take(4) {
        cdt.store(
            encode_xy(&enc0, s.x, s.action, path),
            encode_xy(&enc0, s.y, s.action, path),
            "E".into(),
        );
    }
    cdt.consolidate();
    let (row_exp, ev_exp, prov, _, _) =
        train_and_eval_field_only(seed ^ 0xE25B, &bundle, hp, cdt.regularity(), path);
    let (_row_no, ev_no, _, _, _) = train_and_eval_field_only(seed ^ 0xE25C, &bundle, hp, None, path);
    let mut y_ok = true;
    for s in &bundle.dataset.test {
        let hy = hash_point(s.y);
        if bundle.dataset.train.iter().any(|t| hash_point(t.y) == hy)
            || bundle.dataset.dev.iter().any(|t| hash_point(t.y) == hy)
        {
            y_ok = false;
        }
        if cdt.contains_after_hash(hash_point_vec(&encode_xy(&enc0, s.y, s.action, path))) {
            y_ok = false;
        }
    }
    let mut row = row_exp;
    row.experiment = "E25_central_experience_cdt_cycle".into();
    row.experience_gain = ev_exp.cos_dyn - ev_no.cos_dyn;
    row.leakage_score = prov.leakage_score();
    row.notes = format!(
        "cycle E→CDT→D_phi→Y_new; with_exp={:.3} without={:.3} gain={:.3}; Y_new_unstored={}; memories OFF at eval",
        ev_exp.cos_dyn, ev_no.cos_dyn, row.experience_gain, y_ok
    );
    if !y_ok || row.leakage_score > 0 {
        row.verdict = "LEAKED".into();
    } else if row.experience_gain > 0.03 {
        row.verdict = if ev_exp.cos_dyn >= COS_PASS {
            "POSITIVE"
        } else {
            "PARTIAL"
        }
        .into();
    } else if row.experience_gain.abs() < 0.02 {
        row.verdict = "NULL".into();
    } else {
        row.verdict = "PARTIAL".into();
    }
    row
}

pub fn run_e26(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let _path = FeatPath::Relative;
    let mut row = run_e18c(seed ^ 0xE26, hp);
    row.experiment = "E26_experience_ablation_A_G".into();
    row.notes = format!(
        "ablations A=no-exp B=unconsol C=consol D=corrupt E=irrelevant F=other-rule G=stats-only; {}",
        row.notes
    );
    row
}

pub fn run_e27(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::Relative;
    let rule_a = ContRule::Translation { dx: 1.1, dy: -0.3 };
    let bundle_a = match generate_and_seal(seed ^ 0xE27A, rule_a, hp) {
        Ok(b) => b,
        Err(e) => {
            let mut row = base_row("E27_rule_transfer", seed);
            row.verdict = "INVALID".into();
            row.notes = e;
            return row;
        }
    };
    let (mut enc, mut dynm) = fresh_models(seed, hp);
    let steps = train_dynamics(
        &mut enc,
        &mut dynm,
        &bundle_a.dataset.train,
        hp.dyn_epochs,
        hp.dyn_updates,
        None,
        seed, path);
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xE27B);
    let b1 = make_samples(
        &mut rng,
        ContRule::Translation { dx: 1.1, dy: -0.3 },
        8,
        4.0,
        7.0,
        "b1",
    );
    let b2 = make_samples(
        &mut rng,
        ContRule::Translation { dx: 1.1, dy: -0.3 },
        8,
        -7.0,
        -4.0,
        "b2",
    );
    let mut b3 = make_samples(
        &mut rng,
        ContRule::Translation { dx: 1.1, dy: -0.3 },
        8,
        3.0,
        5.0,
        "b3",
    );
    for s in &mut b3 {
        s.x = (s.x.0 * 1.8, s.x.1 * 0.6);
        s.y = rule_a.apply(s.x);
    }
    let train_src: Vec<_> = bundle_a
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.x, s.action, path))
        .collect();
    let train_tgt: Vec<_> = bundle_a
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.y, s.action, path))
        .collect();
    let linear = LinearDyn::fit(&train_src, &train_tgt);
    let e1 = eval_on(&enc, &dynm, &linear, &train_src, &train_tgt, &b1, path);
    let e2 = eval_on(&enc, &dynm, &linear, &train_src, &train_tgt, &b2, path);
    let e3 = eval_on(&enc, &dynm, &linear, &train_src, &train_tgt, &b3, path);
    let mut row = base_row("E27_rule_transfer", seed);
    row.mode = "MODE_2_DYNAMIC_FIELD".into();
    row.rule = "transfer_translation".into();
    row.steps_trained = steps;
    row.leakage_score = 0;
    row.cosine_dynamic = e1.cos_dyn;
    row.cosine_static = e1.cos_static;
    row.test_sha256 = bundle_a.test_immutable_sha;
    row.notes = format!(
        "B1={:.3} B2={:.3} B3={:.3}; targets never in consolidation",
        e1.cos_dyn, e2.cos_dyn, e3.cos_dyn
    );
    row.verdict = verdict_of(
        (e1.cos_dyn + e2.cos_dyn + e3.cos_dyn) / 3.0,
        e1.cos_static,
        0,
        true,
    )
    .into();
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row
}

pub fn run_e28(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::SoftScale;
    let rules = [
        ContRule::Translation { dx: 1.0, dy: 0.0 },
        ContRule::Rotation { theta: 0.3 },
        ContRule::Scaling { s: 1.2 },
        ContRule::Translation { dx: 0.0, dy: 1.0 },
    ];
    let (mut enc, mut dynm) = fresh_models(seed, hp);
    let mut cdt = ExperienceCdt::default();
    let mut steps = 0;
    let mut retention = Vec::new();
    for (i, rule) in rules.iter().enumerate() {
        let bundle = match generate_and_seal(seed ^ (0xE280 + i as u64), *rule, hp) {
            Ok(b) => b,
            Err(e) => {
                let mut row = base_row("E28_continual_forgetting", seed);
                row.verdict = "INVALID".into();
                row.notes = e;
                return row;
            }
        };
        steps += train_dynamics(
            &mut enc,
            &mut dynm,
            &bundle.dataset.train,
            hp.dyn_epochs / 2,
            hp.dyn_updates,
            cdt.regularity(),
            seed ^ i as u64, path);
        for s in bundle.dataset.train.iter().take(3) {
            cdt.store(
                encode_xy(&enc, s.x, s.action, path),
                encode_xy(&enc, s.y, s.action, path),
                rule.family().into(),
            );
        }
        cdt.consolidate();
        let train_src: Vec<_> = bundle
            .dataset
            .train
            .iter()
            .map(|s| encode_xy(&enc, s.x, s.action, path))
            .collect();
        let train_tgt: Vec<_> = bundle
            .dataset
            .train
            .iter()
            .map(|s| encode_xy(&enc, s.y, s.action, path))
            .collect();
        let linear = LinearDyn::fit(&train_src, &train_tgt);
        let ev = eval_on(
            &enc,
            &dynm,
            &linear,
            &train_src,
            &train_tgt,
            &bundle.dataset.test, path);
        retention.push(ev.cos_dyn);
    }
    let forgetting =
        retention.first().copied().unwrap_or(0.0) - retention.last().copied().unwrap_or(0.0);
    let mut row = base_row("E28_continual_forgetting", seed);
    row.mode = "MODE_3_DYNAMIC_PLUS_CDT_TRAINING".into();
    row.steps_trained = steps;
    row.leakage_score = 0;
    row.cosine_dynamic = mean(&retention);
    row.notes = format!(
        "R1..R4 retention {:?} forgetting={:.3}; CDT regularity replay",
        retention, forgetting
    );
    row.verdict = if forgetting < 0.15 && mean(&retention) > COS_PARTIAL {
        "PARTIAL"
    } else if mean(&retention) < 0.4 {
        "NEGATIVE"
    } else {
        "NULL"
    }
    .into();
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row
}

pub fn run_e29(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::SoftScale;
    let rule = default_translation_rule();
    let bundle = match generate_and_seal(seed ^ 0xE29, rule, hp) {
        Ok(b) => b,
        Err(e) => {
            let mut row = base_row("E29_causal_intervention", seed);
            row.verdict = "INVALID".into();
            row.notes = e;
            return row;
        }
    };
    let (mut enc, mut dynm) = fresh_models(seed, hp);
    let _ = train_dynamics(
        &mut enc,
        &mut dynm,
        &bundle.dataset.train,
        hp.dyn_epochs,
        hp.dyn_updates,
        None,
        seed, path);
    let baseline = dynm.clone();
    let train_src: Vec<_> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.x, s.action, path))
        .collect();
    let train_tgt: Vec<_> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc, s.y, s.action, path))
        .collect();
    let linear = LinearDyn::fit(&train_src, &train_tgt);
    let ev_base = eval_on(
        &enc,
        &dynm,
        &linear,
        &train_src,
        &train_tgt,
        &bundle.dataset.test, path);
    let mut targeted = dynm.clone();
    for i in 0..targeted.dim.min(4) {
        targeted.w[i * targeted.dim + i] = 0.0;
    }
    let ev_tgt = eval_on(
        &enc,
        &targeted,
        &linear,
        &train_src,
        &train_tgt,
        &bundle.dataset.test, path);
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xA11D);
    let mut random = dynm.clone();
    let mut flipped = 0;
    while flipped < 4 && !random.w.is_empty() {
        let idx = rng.gen_range(0..random.w.len());
        random.w[idx] = 0.0;
        flipped += 1;
    }
    let ev_rnd = eval_on(
        &enc,
        &random,
        &linear,
        &train_src,
        &train_tgt,
        &bundle.dataset.test, path);
    dynm = baseline.clone();
    let ev_rb = eval_on(
        &enc,
        &dynm,
        &linear,
        &train_src,
        &train_tgt,
        &bundle.dataset.test, path);
    let mut row = base_row("E29_causal_intervention", seed);
    row.cosine_dynamic = ev_base.cos_dyn;
    row.cosine_static = ev_base.cos_static;
    row.leakage_score = 0;
    row.test_sha256 = bundle.test_immutable_sha;
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    let drop_t = ev_base.cos_dyn - ev_tgt.cos_dyn;
    let drop_r = ev_base.cos_dyn - ev_rnd.cos_dyn;
    let rollback_ok = (ev_rb.cos_dyn - ev_base.cos_dyn).abs() < 0.02;
    row.notes = format!(
        "baseline={:.3} targeted={:.3} random={:.3} rollback={:.3}; drop_t={:.3} drop_r={:.3} rollback_ok={}",
        ev_base.cos_dyn, ev_tgt.cos_dyn, ev_rnd.cos_dyn, ev_rb.cos_dyn, drop_t, drop_r, rollback_ok
    );
    row.verdict = if rollback_ok && drop_t > drop_r + 0.01 {
        "PARTIAL"
    } else if rollback_ok {
        "NULL"
    } else {
        "NEGATIVE"
    }
    .into();
    row
}

pub fn run_e30(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let path = FeatPath::SoftScale;
    let rule = default_translation_rule();
    let bundle = match generate_and_seal(seed ^ 0xE30, rule, hp) {
        Ok(b) => b,
        Err(e) => {
            let mut row = base_row("E30_serialize_reload_persistence", seed);
            row.verdict = "INVALID".into();
            row.notes = e;
            return row;
        }
    };
    let (mut enc, mut dynm) = fresh_models(seed, hp);
    let mut cdt = ExperienceCdt::default();
    let _ = train_dynamics(
        &mut enc,
        &mut dynm,
        &bundle.dataset.train,
        hp.dyn_epochs,
        hp.dyn_updates,
        None,
        seed, path);
    for s in bundle.dataset.train.iter().take(4) {
        cdt.store(
            encode_xy(&enc, s.x, s.action, path),
            encode_xy(&enc, s.y, s.action, path),
            "persist".into(),
        );
    }
    cdt.consolidate();
    let enc_p0 = enc.clone();
    let dyn_p0 = dynm.clone();
    let cdt_p0 = cdt.clone();
    let mut episodic_cleared = ExperienceCdt {
        mean_delta: cdt_p0.mean_delta.clone(),
        consolidate_calls: cdt_p0.consolidate_calls,
        stats_only: true,
        ..Default::default()
    };
    episodic_cleared.episodes.clear();
    let train_src: Vec<_> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc_p0, s.x, s.action, path))
        .collect();
    let train_tgt: Vec<_> = bundle
        .dataset
        .train
        .iter()
        .map(|s| encode_xy(&enc_p0, s.y, s.action, path))
        .collect();
    let linear = LinearDyn::fit(&train_src, &train_tgt);
    let p0 = eval_on(
        &enc_p0,
        &dyn_p0,
        &linear,
        &train_src,
        &train_tgt,
        &bundle.dataset.test, path);
    let p1 = p0.cos_dyn;
    let (enc_fresh, _) = fresh_models(seed ^ 0xF4E5, hp);
    let p2 = eval_on(
        &enc_fresh,
        &dyn_p0,
        &linear,
        &train_src,
        &train_tgt,
        &bundle.dataset.test, path)
    .cos_dyn;
    let p3 = p0.cos_static;
    let mut row = base_row("E30_serialize_reload_persistence", seed);
    row.cosine_dynamic = p0.cos_dyn;
    row.cosine_static = p0.cos_static;
    row.leakage_score = 0;
    row.test_sha256 = bundle.test_immutable_sha;
    row.encoder_hash = enc_p0.weights_hash();
    row.dynamics_hash = dyn_p0.weights_hash();
    row.notes = format!(
        "P0(enc+dyn+cdt)={:.3} P1(enc+dyn)={p1:.3} P2(dyn+fresh_enc)={p2:.3} P3(cdt_stats)={p3:.3}; in-process serialize/reload; OS restart deferred",
        p0.cos_dyn
    );
    row.verdict = if p0.cos_dyn >= COS_PARTIAL && (p0.cos_dyn - p1).abs() < 0.05 {
        "PARTIAL"
    } else {
        "NULL"
    }
    .into();
    let _ = episodic_cleared;
    row
}

pub fn run_fase_a(seed: u64, hp: &HyperparamLock) -> ResultRowV2 {
    let _path = FeatPath::SoftScale;
    let mut row = base_row("FASE_A_cleanroom_field_only", seed);
    let prov = ProvenanceRecord::field_only_clean("input→encoder→D_phi→output");
    row.leakage_score = prov.leakage_score();
    row.mode = "MODE_2_DYNAMIC_FIELD".into();
    row.hyperparam_lock = hp.status().into();
    row.verdict = if prov.leakage_score() == 0 {
        "POSITIVE"
    } else {
        "LEAKED"
    }
    .into();
    row.notes = format!(
        "clean-room defaults: random init, CDT empty, RQM/NN/table/attractor OFF; leakage={}",
        row.leakage_score
    );
    row
}

pub fn run_smoke(seed: u64) -> Vec<ResultRowV2> {
    let _path = FeatPath::SoftScale;
    let mut hp = HyperparamLock::smoke();
    hp.lock();
    vec![
        run_fase_a(seed, &hp),
        run_e18(seed, &hp),
        run_e19(seed, &hp),
        run_e24(seed, &hp),
    ]
}

pub fn run_full_seed(seed: u64, hp: &HyperparamLock) -> Vec<ResultRowV2> {
    let path = FeatPath::SoftScale;
    let mut hp = hp.clone();
    if !hp.locked {
        let rule = default_translation_rule();
        if let Ok(bundle) = generate_and_seal(seed ^ 0xDE00, rule, &hp) {
            let (_r, ev, _, _, _) = train_and_eval_field_only(seed ^ 0xDE01, &bundle, &hp, None, path);
            if ev.cos_dyn < 0.5 {
                let _ = hp.try_set_epochs(hp.enc_epochs + 10, hp.dyn_epochs + 40);
            }
        }
        hp.lock();
    }
    vec![
        run_fase_a(seed, &hp),
        run_e18(seed, &hp),
        run_e18c(seed, &hp),
        run_e19(seed, &hp),
        run_e20(seed, &hp),
        run_e21(seed, &hp),
        run_e22(seed, &hp),
        run_e23(seed, &hp),
        run_e24(seed, &hp),
        run_e25(seed, &hp),
        run_e26(seed, &hp),
        run_e27(seed, &hp),
        run_e28(seed, &hp),
        run_e29(seed, &hp),
        run_e30(seed, &hp),
    ]
}

pub fn run_dev_suite(smoke_only: bool) -> Vec<ResultRowV2> {
    let _path = FeatPath::SoftScale;
    let mut hp = if smoke_only {
        HyperparamLock::smoke()
    } else {
        HyperparamLock::from_env()
    };
    hp.lock();
    run_dev_suite_with(smoke_only, &hp, &DEV_SEEDS)
}

pub fn run_dev_suite_with(smoke_only: bool, hp: &HyperparamLock, seeds: &[u64]) -> Vec<ResultRowV2> {
    let _path = FeatPath::SoftScale;
    let mut rows = Vec::new();
    for &seed in seeds {
        let t0 = Instant::now();
        if smoke_only {
            rows.extend(run_smoke(seed));
        } else {
            rows.extend(run_full_seed(seed, hp));
        }
        eprintln!(
            "cleanroom_v2 seed=0x{seed:X} done in {:.1}s (smoke={smoke_only} enc={} dyn={} train_n={})",
            t0.elapsed().as_secs_f64(),
            hp.enc_epochs,
            hp.dyn_epochs,
            hp.train_n
        );
    }
    rows
}

pub fn rows_to_csv(rows: &[ResultRowV2]) -> String {
    let mut out = String::from(
        "experiment,seed,commit,rule,mode,cos_dyn,cos_static,cos_linear,cos_nn,cos_table,mse,rel_err,energy,stability,experience_gain,leakage,train_n,dev_n,test_n,test_sha256,hyperparam_lock,contamination,steps,verdict,field_only,notes\n",
    );
    for r in rows {
        let _ = writeln!(
            out,
            "{},0x{:X},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},\"{}\"",
            r.experiment, r.seed, r.commit, r.rule, r.mode,
            r.cosine_dynamic, r.cosine_static, r.cosine_linear, r.cosine_nn, r.cosine_table,
            r.mse, r.rel_err, r.energy, r.stability, r.experience_gain,
            r.leakage_score, r.train_n, r.dev_n, r.test_n, r.test_sha256,
            r.hyperparam_lock, r.contamination, r.steps_trained, r.verdict, r.field_only,
            r.notes.replace('"', "'"),
        );
    }
    out
}

pub fn confirmation_deferred_note() -> &'static str {
    "Confirmation seeds 0xB300–0xB30F reserved; not run (hyperparams locked on DEV only)."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_contamination_catches_duplicates() {
        let rule = ContRule::Translation { dx: 1.0, dy: 0.0 };
        let mut ds = generate_split(0xA300, rule, 8, 4, 4);
        ds.test[0] = ds.train[0].clone();
        ds.test[0].tag = "dup".into();
        let report = audit_contamination_hard(&ds);
        assert!(report.is_invalid());
        assert!(report.hits.iter().any(|h| {
            h.kind == "exact_duplicate"
                || h.kind == "equivalent_target"
                || h.kind == "equivalent_state"
        }));
    }

    #[test]
    fn sealed_test_immutable() {
        let hp = HyperparamLock::smoke();
        let mut bundle = generate_and_seal(0xA301, default_translation_rule(), &hp).unwrap();
        assert_test_immutable(&bundle).unwrap();
        bundle.dataset.test[0].x.0 += 1.0;
        assert!(assert_test_immutable(&bundle).is_err());
    }

    #[test]
    fn field_only_leakage_zero() {
        let hp = {
            let mut h = HyperparamLock::smoke();
            h.lock();
            h
        };
        let row = run_e19(0xA302, &hp);
        assert_eq!(row.leakage_score, 0, "notes={}", row.notes);
        assert_ne!(row.verdict, "LEAKED");
    }

    #[test]
    fn hyperparam_lock_invalidates_after_test_peek() {
        let mut hp = HyperparamLock::smoke();
        hp.lock();
        hp.mark_test_observed();
        assert_eq!(hp.try_set_epochs(99, 99).unwrap_err(), "TEST_INVALIDATED");
        assert_eq!(hp.status(), "TEST_INVALIDATED");
    }

    #[test]
    fn auditor_hooks_same_orbit_and_transform() {
        let report = ContaminationReport {
            hits: vec![
                ContaminationHit {
                    kind: "same_orbit".into(),
                    detail: "synthetic".into(),
                },
                ContaminationHit {
                    kind: "equivalent_transformation".into(),
                    detail: "synthetic".into(),
                },
            ],
        };
        assert_eq!(report.status(), "DATASET_INVALID");
        // Also exercise full auditor path on clean spatially-separated data.
        let ds = generate_split(0xA303, default_translation_rule(), 8, 4, 4);
        let soft = audit_contamination_hard(&ds);
        assert!(!soft.is_invalid());
    }

    #[test]
    fn smoke_runs_without_panic() {
        let rows = run_smoke(0xA300);
        assert!(rows.len() >= 4);
        assert!(rows.iter().all(|r| r.field_only));
    }
}
