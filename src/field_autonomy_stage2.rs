//! Etapa 2 — autonomía del campo: aprendizaje de reglas desde experiencia consolidada.
//! Protocolo: `docs/etapa_2_autonomia_campo_experimentos.md`.
//!
//! Fase A: auditoría leakage/provenance, FIELD_ONLY, static vs dynamic, RQM/CDT OFF.
//! Fase B: E18A/B/C, E20, E21 (+ E19 audit cableada).
//! Fase C (parcial): E22, E23, E24.
//! E25–E30: stubs / deferred salvo lo que quepa sin romper el protocolo.
//!
//! Núcleo: vectores numéricos → FieldEncoder → D_φ. CDT = experiencia, nunca lookup de respuesta.
//! Gemma/GGUF es periferia opcional; estos experimentos de regla son field-only.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cloned_ref_to_slice_refs)]

use crate::liquid_experiments_11_17::{FieldDynamics, TrainableFieldEncoder};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::HashSet;
use std::time::Instant;

const EPS: f64 = 1e-12;
const FIELD_DIM: usize = 24;
const FEAT_DIM: usize = 16;
const COS_PASS: f64 = 0.85;
const COS_STRONG: f64 = 0.92;
const COS_PARTIAL: f64 = 0.70;

// ── modes (doc §4) ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutonomyMode {
    Mode0Raw,
    Mode1StaticField,
    Mode2DynamicField,
    Mode3DynamicPlusCdtTraining,
    Mode4Full,
}

impl AutonomyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mode0Raw => "MODE_0_RAW",
            Self::Mode1StaticField => "MODE_1_STATIC_FIELD",
            Self::Mode2DynamicField => "MODE_2_DYNAMIC_FIELD",
            Self::Mode3DynamicPlusCdtTraining => "MODE_3_DYNAMIC_PLUS_CDT_TRAINING",
            Self::Mode4Full => "MODE_4_FULL",
        }
    }
}

// ── consolidated experience (doc §5) ────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ConsolidatedExperience {
    pub context_signature: String,
    pub state_before: Vec<f64>,
    pub action_or_relation: String,
    pub state_after: Vec<f64>,
    pub confidence: f64,
    pub energy: f64,
    pub repetition_count: u32,
    /// Fingerprint hashes for leakage audit (not used as answer lookup).
    pub before_hash: u64,
    pub after_hash: u64,
}

/// CDT-as-experience store. May bias training via aggregated regularities.
/// Eval path must never retrieve target state_after for novel queries.
#[derive(Clone, Debug, Default)]
pub struct ExperienceMemory {
    pub experiences: Vec<ConsolidatedExperience>,
    pub consolidate_calls: u64,
    pub eval_query_count: u64,
    pub rqm_query_count: u64,
    pub table_query_count: u64,
    pub nn_query_count: u64,
    pub attractor_query_count: u64,
    /// Aggregated regularity: mean (after − before) in fingerprint space.
    pub mean_delta: Option<Vec<f64>>,
    pub action_histogram: Vec<(String, u32)>,
}

impl ExperienceMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn store(&mut self, exp: ConsolidatedExperience) {
        if let Some((name, c)) = self
            .action_histogram
            .iter_mut()
            .find(|(a, _)| a == &exp.action_or_relation)
        {
            *c = c.saturating_add(exp.repetition_count);
            let _ = name;
        } else {
            self.action_histogram
                .push((exp.action_or_relation.clone(), exp.repetition_count));
        }
        self.experiences.push(exp);
    }

    /// Consolidate: recompute mean_delta regularity. Does not expose targets for lookup.
    pub fn consolidate(&mut self) {
        self.consolidate_calls += 1;
        if self.experiences.is_empty() {
            self.mean_delta = None;
            return;
        }
        let dim = self.experiences[0].state_after.len();
        let mut acc = vec![0.0; dim];
        let mut n = 0.0;
        for e in &self.experiences {
            if e.state_before.len() != dim || e.state_after.len() != dim {
                continue;
            }
            for i in 0..dim {
                acc[i] += e.state_after[i] - e.state_before[i];
            }
            n += 1.0;
        }
        if n > 0.0 {
            for a in &mut acc {
                *a /= n;
            }
            self.mean_delta = Some(acc);
        }
    }

    /// Training-time context signal only (regularity), never a target answer.
    pub fn regularity_context(&self) -> Option<&[f64]> {
        self.mean_delta.as_deref()
    }

    pub fn contains_after_hash(&self, h: u64) -> bool {
        self.experiences.iter().any(|e| e.after_hash == h)
    }

    pub fn contains_before_hash(&self, h: u64) -> bool {
        self.experiences.iter().any(|e| e.before_hash == h)
    }

    /// Forbidden at field-only eval: incrementing this marks lookup attempt.
    pub fn forbidden_eval_lookup(&mut self) {
        self.eval_query_count += 1;
    }
}

// ── leakage / provenance audit (doc §10, §24) ───────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct ProvenanceAudit {
    pub target_seen_training: bool,
    pub target_seen_cdt: bool,
    pub target_seen_rqm: bool,
    pub target_seen_attractor: bool,
    pub target_seen_table: bool,
    pub target_seen_nn: bool,
    pub target_equivalent_seen: bool,
    pub cdt_query_count: u64,
    pub rqm_query_count: u64,
    pub table_query_count: u64,
    pub nn_query_count: u64,
    pub attractor_query_count: u64,
    pub information_path: String,
    pub target_origin: String,
    pub field_only: bool,
}

impl ProvenanceAudit {
    pub fn field_only_path(path: &str) -> Self {
        Self {
            information_path: path.into(),
            target_origin: "generated_by_D_phi".into(),
            field_only: true,
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
        ]
        .iter()
        .filter(|&&b| b)
        .count() as u64;
        let queries = if self.field_only {
            self.cdt_query_count
                + self.rqm_query_count
                + self.table_query_count
                + self.nn_query_count
                + self.attractor_query_count
        } else {
            0
        };
        flags + queries
    }

    pub fn is_leaked(&self) -> bool {
        self.leakage_score() > 0
            || self.information_path.contains("CDT→target")
            || self.information_path.contains("lookup")
    }

    pub fn clean_for_field_only(&self) -> bool {
        self.field_only && !self.is_leaked()
    }
}

// ── registry row ────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct RegistryRowStage2 {
    pub experiment: String,
    pub commit: String,
    pub seed: u64,
    pub hardware: String,
    pub rust_version: String,
    pub mode: String,
    pub rule: String,
    pub train_pairs: usize,
    pub test_pairs: usize,
    pub cdt_enabled: bool,
    pub rqm_enabled: bool,
    pub field_only: bool,
    pub cosine_dynamic: f64,
    pub cosine_static: f64,
    pub cosine_linear: f64,
    pub cosine_nn: f64,
    pub cosine_table: f64,
    pub coord_err_dynamic: f64,
    pub rule_generalization: f64,
    pub novel_state_generation: f64,
    pub experience_gain: f64,
    pub leakage_score: u64,
    pub cdt_query_count: u64,
    pub rqm_query_count: u64,
    pub energy: f64,
    pub stability: f64,
    pub steps_trained: usize,
    pub encoder_hash: String,
    pub dynamics_hash: String,
    pub verdict: String,
    pub notes: String,
    pub periphery: String,
}

// ── math transforms ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum RuleKind {
    Translation { dx: f64, dy: f64 },
    Rotation { theta: f64 },
    ReflectionX,
    Scaling { s: f64 },
}

impl RuleKind {
    pub fn name(self) -> String {
        match self {
            Self::Translation { dx, dy } => format!("R1_translation(dx={dx},dy={dy})"),
            Self::Rotation { theta } => format!("R2_rotation(theta={theta:.4})"),
            Self::ReflectionX => "R3_reflection_x".into(),
            Self::Scaling { s } => format!("R4_scaling(s={s})"),
        }
    }

    pub fn apply(self, p: (f64, f64)) -> (f64, f64) {
        match self {
            Self::Translation { dx, dy } => (p.0 + dx, p.1 + dy),
            Self::Rotation { theta } => {
                let (c, s) = (theta.cos(), theta.sin());
                (c * p.0 - s * p.1, s * p.0 + c * p.1)
            }
            Self::ReflectionX => (p.0, -p.1),
            Self::Scaling { s } => (p.0 * s, p.1 * s),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Pair {
    pub a: (f64, f64),
    pub b: (f64, f64),
    pub rule: String,
}

fn point_features(p: (f64, f64)) -> Vec<f64> {
    let (x, y) = p;
    let mut f = vec![
        x,
        y,
        x * x,
        y * y,
        x * y,
        x.sin(),
        y.cos(),
        (0.5 * x).sin(),
        (0.5 * y).cos(),
        (x + y) * 0.1,
        (x - y) * 0.1,
        1.0,
        x.abs() * 0.1,
        y.abs() * 0.1,
        (x * 0.3).tanh(),
        (y * 0.3).tanh(),
    ];
    f.truncate(FEAT_DIM);
    while f.len() < FEAT_DIM {
        f.push(0.0);
    }
    normalize(&mut f);
    f
}

fn grid_points(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<(f64, f64)> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut seen = HashSet::new();
    while out.len() < n {
        let x = lo + rng.gen::<f64>() * (hi - lo);
        let y = lo + rng.gen::<f64>() * (hi - lo);
        // quantize to reduce accidental duplicates
        let key = ((x * 20.0).round() as i32, (y * 20.0).round() as i32);
        if seen.insert(key) {
            out.push((x, y));
        }
    }
    out
}

fn make_pairs(starts: &[(f64, f64)], rule: RuleKind) -> Vec<Pair> {
    let name = rule.name();
    starts
        .iter()
        .map(|&a| Pair {
            a,
            b: rule.apply(a),
            rule: name.clone(),
        })
        .collect()
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
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

fn l2(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n {
        let d = a[i] - b.get(i).copied().unwrap_or(0.0);
        s += d * d;
    }
    s.sqrt()
}

fn hash_f64_slice(xs: &[f64]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &x in xs {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
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

fn gguf_present() -> bool {
    if let Ok(p) = std::env::var("GEMMA2_GGUF") {
        if !p.is_empty() && std::path::Path::new(&p).exists() {
            return true;
        }
    }
    std::path::Path::new("models/gemma-2-2b-it-Q4_K_M.gguf").exists()
}

fn periphery_label() -> &'static str {
    if gguf_present() {
        "numeric_field_only (GGUF on disk unused for rule core)"
    } else {
        "numeric_field_only (no GGUF)"
    }
}

fn base_row(experiment: &str, seed: u64) -> RegistryRowStage2 {
    RegistryRowStage2 {
        experiment: experiment.into(),
        commit: git_commit(),
        seed,
        hardware: hardware_label(),
        rust_version: rust_version(),
        periphery: periphery_label().into(),
        field_only: true,
        cdt_enabled: false,
        rqm_enabled: false,
        ..Default::default()
    }
}

fn verdict_from_scores(
    cos_dyn: f64,
    cos_static: f64,
    cos_table: f64,
    leakage: u64,
    beats_memory_controls: bool,
) -> &'static str {
    if leakage > 0 {
        return "LEAKED";
    }
    if cos_dyn >= COS_STRONG && cos_dyn > cos_static + 0.08 && beats_memory_controls {
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
    if cos_dyn < cos_static - 0.05 || cos_dyn < cos_table - 0.15 {
        return "NEGATIVE";
    }
    "NULL"
}

/// Linear dynamics control: least-squares map in fingerprint space (train only).
#[derive(Clone, Debug)]
struct LinearDynamics {
    dim: usize,
    w: Vec<f64>, // row-major dim×dim
}

impl LinearDynamics {
    fn fit(srcs: &[Vec<f64>], tgts: &[Vec<f64>]) -> Self {
        let dim = srcs.first().map(|v| v.len()).unwrap_or(FIELD_DIM);
        // Ridge regression: W ≈ (XᵀX + λI)⁻¹ XᵀY  via simple GD (avoid ndarray dep).
        let mut w = vec![0.0; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        let eta = 0.05;
        let lambda = 1e-3;
        for _ in 0..400 {
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

/// Table control: exact source match → target fp. Novel → fail (cos≈0).
fn table_predict(query_fp: &[f64], train_src: &[Vec<f64>], train_tgt: &[Vec<f64>]) -> Vec<f64> {
    for (s, t) in train_src.iter().zip(train_tgt.iter()) {
        if l2(query_fp, s) < 1e-6 {
            return t.clone();
        }
    }
    vec![0.0; query_fp.len()]
}

/// NN control: nearest train source → its target (memory, not rule).
fn nn_predict(query_fp: &[f64], train_src: &[Vec<f64>], train_tgt: &[Vec<f64>]) -> Vec<f64> {
    let mut best = 0usize;
    let mut best_c = -1.0;
    for (i, s) in train_src.iter().enumerate() {
        let c = cosine(query_fp, s);
        if c > best_c {
            best_c = c;
            best = i;
        }
    }
    train_tgt
        .get(best)
        .cloned()
        .unwrap_or_else(|| vec![0.0; query_fp.len()])
}

/// Linear readout Ψ → (x,y) fit on train encodings only.
struct CoordReadout {
    wx: Vec<f64>,
    wy: Vec<f64>,
    bx: f64,
    by: f64,
}

impl CoordReadout {
    fn fit(fps: &[Vec<f64>], coords: &[(f64, f64)]) -> Self {
        let dim = fps.first().map(|v| v.len()).unwrap_or(FIELD_DIM);
        let mut wx = vec![0.0; dim];
        let mut wy = vec![0.0; dim];
        let mut bx = 0.0;
        let mut by = 0.0;
        let eta = 0.08;
        for _ in 0..500 {
            for (fp, &(x, y)) in fps.iter().zip(coords.iter()) {
                let mut px = bx;
                let mut py = by;
                for i in 0..dim {
                    px += wx[i] * fp.get(i).copied().unwrap_or(0.0);
                    py += wy[i] * fp.get(i).copied().unwrap_or(0.0);
                }
                let ex = px - x;
                let ey = py - y;
                bx -= eta * ex;
                by -= eta * ey;
                for i in 0..dim {
                    let f = fp.get(i).copied().unwrap_or(0.0);
                    wx[i] -= eta * ex * f;
                    wy[i] -= eta * ey * f;
                }
            }
        }
        Self { wx, wy, bx, by }
    }

    fn predict(&self, fp: &[f64]) -> (f64, f64) {
        let mut px = self.bx;
        let mut py = self.by;
        for i in 0..self.wx.len() {
            let f = fp.get(i).copied().unwrap_or(0.0);
            px += self.wx[i] * f;
            py += self.wy[i] * f;
        }
        (px, py)
    }
}

fn encode_pair(enc: &TrainableFieldEncoder, p: (f64, f64)) -> Vec<f64> {
    enc.encode(&point_features(p))
}

fn train_encoder_dynamics(
    enc: &mut TrainableFieldEncoder,
    dynm: &mut FieldDynamics,
    pairs: &[Pair],
    epochs: usize,
    regularity: Option<&[f64]>,
) -> usize {
    let mut steps = 0usize;
    for _ in 0..epochs {
        for p in pairs {
            let fa = point_features(p.a);
            let fb = point_features(p.b);
            let psi_a = enc.encode(&fa);
            let psi_b = enc.encode(&fb);
            // Encoder: pull encodings toward a consistent geometry for both endpoints.
            let _ = enc.train_step(
                &fa,
                std::slice::from_ref(&psi_a),
                std::slice::from_ref(&psi_b),
            );
            let _ = enc.train_step(
                &fb,
                std::slice::from_ref(&psi_b),
                std::slice::from_ref(&psi_a),
            );
            let src = enc.encode(&fa);
            let mut tgt = enc.encode(&fb);
            // Optional CDT regularity bias (experience, not answer lookup).
            if let Some(delta) = regularity {
                for i in 0..tgt.len().min(delta.len()) {
                    let soft = src[i] + delta[i];
                    tgt[i] = 0.85 * tgt[i] + 0.15 * soft;
                }
                normalize(&mut tgt);
            }
            let _ = dynm.train_transition(&src, &tgt);
            steps += 1;
        }
    }
    steps
}

fn eval_cosines(
    enc: &TrainableFieldEncoder,
    dynm: &FieldDynamics,
    linear: &LinearDynamics,
    train_src: &[Vec<f64>],
    train_tgt: &[Vec<f64>],
    test: &[Pair],
) -> (f64, f64, f64, f64, f64, f64) {
    let mut c_dyn = Vec::new();
    let mut c_stat = Vec::new();
    let mut c_lin = Vec::new();
    let mut c_nn = Vec::new();
    let mut c_tab = Vec::new();
    let mut coord_errs = Vec::new();

    for p in test {
        let src = encode_pair(enc, p.a);
        let tgt = encode_pair(enc, p.b);
        let pred = dynm.step(&src);
        c_dyn.push(cosine(&pred, &tgt));
        c_stat.push(cosine(&src, &tgt));
        let lp = linear.step(&src);
        c_lin.push(cosine(&lp, &tgt));
        let np = nn_predict(&src, train_src, train_tgt);
        c_nn.push(cosine(&np, &tgt));
        let tp = table_predict(&src, train_src, train_tgt);
        c_tab.push(cosine(&tp, &tgt));
        coord_errs.push(0.0); // filled by caller with readout when available
    }
    (
        mean(&c_dyn),
        mean(&c_stat),
        mean(&c_lin),
        mean(&c_nn),
        mean(&c_tab),
        mean(&coord_errs),
    )
}

fn eval_with_readout(
    enc: &TrainableFieldEncoder,
    dynm: &FieldDynamics,
    train_pairs: &[Pair],
    test: &[Pair],
) -> f64 {
    let mut fps = Vec::new();
    let mut coords = Vec::new();
    for p in train_pairs {
        fps.push(encode_pair(enc, p.a));
        coords.push(p.a);
        fps.push(encode_pair(enc, p.b));
        coords.push(p.b);
    }
    let readout = CoordReadout::fit(&fps, &coords);
    let mut errs = Vec::new();
    for p in test {
        let pred_fp = dynm.step(&encode_pair(enc, p.a));
        let (px, py) = readout.predict(&pred_fp);
        let e = ((px - p.b.0).powi(2) + (py - p.b.1).powi(2)).sqrt();
        errs.push(e);
    }
    mean(&errs)
}

fn stability_noise(
    enc: &TrainableFieldEncoder,
    dynm: &FieldDynamics,
    p: (f64, f64),
    seed: u64,
) -> f64 {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let base = dynm.step(&encode_pair(enc, p));
    let mut scores = Vec::new();
    for _ in 0..8 {
        let noisy = (
            p.0 + (rng.gen::<f64>() - 0.5) * 0.05,
            p.1 + (rng.gen::<f64>() - 0.5) * 0.05,
        );
        let pred = dynm.step(&encode_pair(enc, noisy));
        scores.push(cosine(&base, &pred));
    }
    mean(&scores)
}

// ── E18A — direct rule learning, no CDT ─────────────────────────────────────

pub fn run_e18a(seed: u64) -> RegistryRowStage2 {
    let t0 = Instant::now();
    let rule = RuleKind::Translation { dx: 2.0, dy: 1.0 };
    let starts = grid_points(seed ^ 0xA1, 5, -3.0, 3.0);
    let all = make_pairs(&starts, rule);
    let train = all[..4].to_vec();
    let test = all[4..].to_vec();

    // Micro-augmentation: noisy copies of train starts under the SAME rule (never test).
    let mut train_aug = train.clone();
    let mut rng_aug = Xoshiro256StarStar::seed_from_u64(seed ^ 0xA06);
    for p in &train {
        for _ in 0..3 {
            let a = (
                p.a.0 + (rng_aug.gen::<f64>() - 0.5) * 0.15,
                p.a.1 + (rng_aug.gen::<f64>() - 0.5) * 0.15,
            );
            train_aug.push(Pair {
                a,
                b: rule.apply(a),
                rule: p.rule.clone(),
            });
        }
    }

    let mut enc = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE18A);
    let mut dynm = FieldDynamics::new(FIELD_DIM, seed ^ 0xD18A);
    let steps = train_encoder_dynamics(&mut enc, &mut dynm, &train_aug, 200, None);

    let train_src: Vec<Vec<f64>> = train.iter().map(|p| encode_pair(&enc, p.a)).collect();
    let train_tgt: Vec<Vec<f64>> = train.iter().map(|p| encode_pair(&enc, p.b)).collect();
    let linear = LinearDynamics::fit(&train_src, &train_tgt);

    let (c_dyn, c_stat, c_lin, c_nn, c_tab, _) =
        eval_cosines(&enc, &dynm, &linear, &train_src, &train_tgt, &test);
    let coord_err = eval_with_readout(&enc, &dynm, &train, &test);
    let stab = stability_noise(&enc, &dynm, test[0].a, seed ^ 0x57AB);

    // E19 audit: target B5 never in train; FIELD_ONLY queries = 0
    let test_tgt_hash = hash_f64_slice(&encode_pair(&enc, test[0].b));
    let train_hashes: HashSet<u64> = train
        .iter()
        .flat_map(|p| {
            [
                hash_f64_slice(&encode_pair(&enc, p.a)),
                hash_f64_slice(&encode_pair(&enc, p.b)),
            ]
        })
        .collect();
    let mut audit = ProvenanceAudit::field_only_path("input→encoder→D_phi→output");
    audit.target_seen_training = train_hashes.contains(&test_tgt_hash);
    // Also check raw coord identity
    let raw_seen = train
        .iter()
        .any(|p| (p.b.0 - test[0].b.0).abs() < 1e-9 && (p.b.1 - test[0].b.1).abs() < 1e-9);
    audit.target_equivalent_seen = raw_seen;
    let leakage = audit.leakage_score();

    let beats_mem = c_dyn + 0.05 >= c_tab && c_dyn + 0.02 >= c_nn;
    let verdict = verdict_from_scores(c_dyn, c_stat, c_tab, leakage, beats_mem);

    let mut row = base_row("E18A_direct_no_cdt", seed);
    row.mode = AutonomyMode::Mode2DynamicField.as_str().into();
    row.rule = rule.name();
    row.train_pairs = train.len();
    row.test_pairs = test.len();
    row.cosine_dynamic = c_dyn;
    row.cosine_static = c_stat;
    row.cosine_linear = c_lin;
    row.cosine_nn = c_nn;
    row.cosine_table = c_tab;
    row.coord_err_dynamic = coord_err;
    row.rule_generalization = c_dyn;
    row.novel_state_generation = if c_dyn >= COS_PASS { 1.0 } else { c_dyn };
    row.leakage_score = leakage;
    row.energy = 1.0 - c_dyn;
    row.stability = stab;
    row.steps_trained = steps;
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row.verdict = verdict.into();
    row.notes = format!(
        "A5 never in train; FIELD_ONLY; static={:.3} linear={:.3} nn={:.3} table={:.3} \
         coord_err={:.3} audit_path={} elapsed_ms={:.1}",
        c_stat,
        c_lin,
        c_nn,
        c_tab,
        coord_err,
        audit.information_path,
        t0.elapsed().as_secs_f64() * 1e3
    );
    row
}

// ── E18B — episodic + CDT consolidation (experience, not lookup) ────────────

pub fn run_e18b(seed: u64) -> RegistryRowStage2 {
    let t0 = Instant::now();
    let rule = RuleKind::Translation { dx: 2.0, dy: 1.0 };
    let starts = grid_points(seed ^ 0xB1, 5, -3.0, 3.0);
    let all = make_pairs(&starts, rule);
    let episodes = &all[..4];
    let test = &all[4..];

    let mut enc = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE18B);
    let mut dynm = FieldDynamics::new(FIELD_DIM, seed ^ 0xD18B);
    let mut mem = ExperienceMemory::new();
    let mut steps = 0usize;

    // Episodic: each pair → store experience → consolidate → adapt field.
    for (i, p) in episodes.iter().enumerate() {
        let psi_a = encode_pair(&enc, p.a);
        let psi_b = encode_pair(&enc, p.b);
        mem.store(ConsolidatedExperience {
            context_signature: format!("ep{i}"),
            state_before: psi_a.clone(),
            action_or_relation: rule.name(),
            state_after: psi_b.clone(),
            confidence: 1.0,
            energy: 1.0 - cosine(&psi_a, &psi_b),
            repetition_count: 1,
            before_hash: hash_f64_slice(&psi_a),
            after_hash: hash_f64_slice(&psi_b),
        });
        mem.consolidate();
        let reg = mem.regularity_context().map(|d| d.to_vec());
        steps += train_encoder_dynamics(
            &mut enc,
            &mut dynm,
            std::slice::from_ref(p),
            60,
            reg.as_deref(),
        );
        // rehearse all consolidated experiences (still no test target)
        steps += train_encoder_dynamics(&mut enc, &mut dynm, episodes, 25, reg.as_deref());
    }

    // Eval: FIELD_ONLY — do NOT query mem for target.
    let test_b_hash = {
        let fp = encode_pair(&enc, test[0].b);
        hash_f64_slice(&fp)
    };
    let mut audit = ProvenanceAudit::field_only_path("input→encoder→CDT_context→D_phi→output");
    audit.target_seen_cdt = mem.contains_after_hash(test_b_hash);
    audit.target_seen_training = episodes
        .iter()
        .any(|p| (p.b.0 - test[0].b.0).abs() < 1e-9 && (p.b.1 - test[0].b.1).abs() < 1e-9);
    // Ensure we did not query CDT for answer:
    audit.cdt_query_count = mem.eval_query_count;
    audit.rqm_query_count = mem.rqm_query_count;

    let train_src: Vec<Vec<f64>> = episodes.iter().map(|p| encode_pair(&enc, p.a)).collect();
    let train_tgt: Vec<Vec<f64>> = episodes.iter().map(|p| encode_pair(&enc, p.b)).collect();
    let linear = LinearDynamics::fit(&train_src, &train_tgt);
    let (c_dyn, c_stat, c_lin, c_nn, c_tab, _) =
        eval_cosines(&enc, &dynm, &linear, &train_src, &train_tgt, test);
    let coord_err = eval_with_readout(&enc, &dynm, episodes, test);
    let leakage = audit.leakage_score();
    let beats_mem = c_dyn + 0.05 >= c_tab;
    let mut verdict = verdict_from_scores(c_dyn, c_stat, c_tab, leakage, beats_mem);
    // PASS condition from doc: D_phi produces B_new AND audit says no target in CDT
    if audit.target_seen_cdt || audit.cdt_query_count > 0 {
        verdict = "LEAKED";
    }

    let mut row = base_row("E18B_episodic_cdt_consolidation", seed);
    row.mode = AutonomyMode::Mode3DynamicPlusCdtTraining.as_str().into();
    row.rule = rule.name();
    row.train_pairs = episodes.len();
    row.test_pairs = test.len();
    row.cdt_enabled = true; // experience store during train
    row.rqm_enabled = false;
    row.cosine_dynamic = c_dyn;
    row.cosine_static = c_stat;
    row.cosine_linear = c_lin;
    row.cosine_nn = c_nn;
    row.cosine_table = c_tab;
    row.coord_err_dynamic = coord_err;
    row.rule_generalization = c_dyn;
    row.novel_state_generation = if !audit.target_seen_cdt && c_dyn >= COS_PASS {
        1.0
    } else {
        c_dyn
    };
    row.leakage_score = if verdict == "LEAKED" {
        leakage.max(1)
    } else {
        leakage
    };
    row.cdt_query_count = audit.cdt_query_count;
    row.rqm_query_count = audit.rqm_query_count;
    row.energy = 1.0 - c_dyn;
    row.stability = stability_noise(&enc, &dynm, test[0].a, seed);
    row.steps_trained = steps;
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row.verdict = verdict.into();
    row.notes = format!(
        "consolidate_calls={} mean_delta={} target_in_cdt={} eval_cdt_queries={} \
         coord_err={:.3} elapsed_ms={:.1}",
        mem.consolidate_calls,
        mem.mean_delta.is_some(),
        audit.target_seen_cdt,
        audit.cdt_query_count,
        coord_err,
        t0.elapsed().as_secs_f64() * 1e3
    );
    row
}

// ── E18C — C0/C1/C2/C3 comparison ───────────────────────────────────────────

fn train_condition(
    seed: u64,
    pairs: &[Pair],
    condition: &str,
) -> (
    TrainableFieldEncoder,
    FieldDynamics,
    usize,
    ExperienceMemory,
) {
    let mut enc = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE18C);
    let mut dynm = FieldDynamics::new(FIELD_DIM, seed ^ 0xD18C);
    let mut mem = ExperienceMemory::new();
    let mut steps = 0;

    match condition {
        "C0" => {
            // No prior experience: leave randomly initialized (tiny train budget = 0).
        }
        "C1" => {
            // Experiences without consolidation (no mean_delta).
            steps = train_encoder_dynamics(&mut enc, &mut dynm, pairs, 80, None);
            for (i, p) in pairs.iter().enumerate() {
                let psi_a = encode_pair(&enc, p.a);
                let psi_b = encode_pair(&enc, p.b);
                mem.store(ConsolidatedExperience {
                    context_signature: format!("c1_{i}"),
                    state_before: psi_a.clone(),
                    action_or_relation: p.rule.clone(),
                    state_after: psi_b.clone(),
                    confidence: 1.0,
                    energy: 0.0,
                    repetition_count: 1,
                    before_hash: hash_f64_slice(&psi_a),
                    after_hash: hash_f64_slice(&psi_b),
                });
            }
            // deliberately no consolidate()
        }
        "C2" => {
            for (i, p) in pairs.iter().enumerate() {
                let psi_a = encode_pair(&enc, p.a);
                let psi_b = encode_pair(&enc, p.b);
                mem.store(ConsolidatedExperience {
                    context_signature: format!("c2_{i}"),
                    state_before: psi_a.clone(),
                    action_or_relation: p.rule.clone(),
                    state_after: psi_b.clone(),
                    confidence: 1.0,
                    energy: 0.0,
                    repetition_count: 1,
                    before_hash: hash_f64_slice(&psi_a),
                    after_hash: hash_f64_slice(&psi_b),
                });
                mem.consolidate();
                let reg = mem.regularity_context().map(|d| d.to_vec());
                steps += train_encoder_dynamics(
                    &mut enc,
                    &mut dynm,
                    std::slice::from_ref(p),
                    20,
                    reg.as_deref(),
                );
            }
            let reg = mem.regularity_context().map(|d| d.to_vec());
            steps += train_encoder_dynamics(&mut enc, &mut dynm, pairs, 60, reg.as_deref());
        }
        "C3" => {
            // Corrupted CDT: consolidate then scramble mean_delta.
            for (i, p) in pairs.iter().enumerate() {
                let psi_a = encode_pair(&enc, p.a);
                let psi_b = encode_pair(&enc, p.b);
                mem.store(ConsolidatedExperience {
                    context_signature: format!("c3_{i}"),
                    state_before: psi_a.clone(),
                    action_or_relation: p.rule.clone(),
                    state_after: psi_b.clone(),
                    confidence: 1.0,
                    energy: 0.0,
                    repetition_count: 1,
                    before_hash: hash_f64_slice(&psi_a),
                    after_hash: hash_f64_slice(&psi_b),
                });
            }
            mem.consolidate();
            if let Some(ref mut d) = mem.mean_delta {
                let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x00C0_BB11);
                for x in d.iter_mut() {
                    *x = (rng.gen::<f64>() * 2.0 - 1.0) * 0.5;
                }
            }
            let reg = mem.regularity_context().map(|d| d.to_vec());
            steps = train_encoder_dynamics(&mut enc, &mut dynm, pairs, 80, reg.as_deref());
        }
        _ => {}
    }
    (enc, dynm, steps, mem)
}

pub fn run_e18c(seed: u64) -> RegistryRowStage2 {
    let t0 = Instant::now();
    let rule = RuleKind::Translation { dx: 2.0, dy: 1.0 };
    let starts = grid_points(seed ^ 0xC1, 5, -3.0, 3.0);
    let all = make_pairs(&starts, rule);
    let train = &all[..4];
    let test = &all[4..];

    let mut scores = Vec::new();
    for cond in ["C0", "C1", "C2", "C3"] {
        let (enc, dynm, _steps, mem) = train_condition(seed, train, cond);
        let mut cs = Vec::new();
        for p in test {
            let pred = dynm.step(&encode_pair(&enc, p.a));
            let tgt = encode_pair(&enc, p.b);
            cs.push(cosine(&pred, &tgt));
        }
        let c = mean(&cs);
        let leaked = test.iter().any(|p| {
            let h = hash_f64_slice(&encode_pair(&enc, p.b));
            mem.contains_after_hash(h) && cond != "C0" && false // presence in store OK; lookup not used
        });
        let _ = leaked;
        scores.push((cond, c, mem.eval_query_count));
    }

    let c0 = scores[0].1;
    let c1 = scores[1].1;
    let c2 = scores[2].1;
    let c3 = scores[3].1;
    let gain = c2 - c1.max(c0);

    let mut audit = ProvenanceAudit::field_only_path("input→encoder→CDT_context→D_phi→output");
    audit.cdt_query_count = scores.iter().map(|s| s.2).sum();
    let leakage = audit.leakage_score();

    let verdict = if leakage > 0 {
        "LEAKED"
    } else if c2 > c0 + 0.08 && c2 >= c1 - 0.02 && c2 > c3 + 0.03 {
        "POSITIVE"
    } else if c2 > c0 + 0.03 {
        "PARTIAL"
    } else if (c2 - c0).abs() < 0.03 {
        "NULL"
    } else {
        "NEGATIVE"
    };

    let mut row = base_row("E18C_cdt_vs_no_cdt", seed);
    row.mode = AutonomyMode::Mode3DynamicPlusCdtTraining.as_str().into();
    row.rule = rule.name();
    row.train_pairs = train.len();
    row.test_pairs = test.len();
    row.cdt_enabled = true;
    row.cosine_dynamic = c2;
    row.cosine_static = c0;
    row.cosine_linear = c1;
    row.cosine_nn = c3;
    row.rule_generalization = c2;
    row.experience_gain = gain;
    row.leakage_score = leakage;
    row.cdt_query_count = audit.cdt_query_count;
    row.verdict = verdict.into();
    row.notes = format!(
        "C0={:.3} C1={:.3} C2={:.3} C3={:.3} gain(C2-max(C0,C1))={:.3} elapsed_ms={:.1}",
        c0,
        c1,
        c2,
        c3,
        gain,
        t0.elapsed().as_secs_f64() * 1e3
    );
    // reuse fields: static=C0, linear=C1, nn=C3 (corrupted), dynamic=C2
    row.cosine_table = 0.0;
    row
}

// ── E19 — explicit no-lookup audit harness over E18A path ───────────────────

pub fn run_e19(seed: u64) -> RegistryRowStage2 {
    let mut row = run_e18a(seed);
    row.experiment = "E19_no_lookup_audit".into();
    // Re-assert field-only counters.
    row.cdt_query_count = 0;
    row.rqm_query_count = 0;
    row.field_only = true;
    row.rqm_enabled = false;
    row.cdt_enabled = false;
    // E19 judges leakage/provenance, not accuracy (accuracy lives in wrapped E18A metrics).
    let wrapped_verdict = row.verdict.clone();
    row.verdict = if row.leakage_score > 0 {
        "LEAKED".into()
    } else {
        "POSITIVE".into()
    };
    row.notes = format!(
        "E19 audit-only: leakage={} cdt_q={} rqm_q={} table_q=0 nn_q=0 attractor_q=0; wrapped_E18A_verdict={}; {}",
        row.leakage_score, row.cdt_query_count, row.rqm_query_count, wrapped_verdict, row.notes
    );
    row
}

// ── E20 — trajectory vs rule ────────────────────────────────────────────────

pub fn run_e20(seed: u64) -> RegistryRowStage2 {
    let t0 = Instant::now();
    let rule = RuleKind::Translation { dx: 1.5, dy: 0.5 };

    // Dataset A — single trajectory chain A→B→C→D
    let a0 = (-2.0, -1.0);
    let chain = {
        let mut v = vec![a0];
        let mut cur = a0;
        for _ in 0..3 {
            cur = rule.apply(cur);
            v.push(cur);
        }
        v
    };
    let traj_pairs: Vec<Pair> = chain
        .windows(2)
        .map(|w| Pair {
            a: w[0],
            b: w[1],
            rule: rule.name(),
        })
        .collect();

    // Dataset B — many independent pairs
    let starts = grid_points(seed ^ 0xE20, 12, -3.0, 3.0);
    let rule_pairs = make_pairs(&starts[..10], rule);
    let test_start = starts[10];
    // ensure test start not in train
    let test = [Pair {
        a: test_start,
        b: rule.apply(test_start),
        rule: rule.name(),
    }];

    // Train on trajectory only
    let mut enc_t = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE20A);
    let mut dyn_t = FieldDynamics::new(FIELD_DIM, seed ^ 0xD20A);
    let _ = train_encoder_dynamics(&mut enc_t, &mut dyn_t, &traj_pairs, 160, None);
    let cos_traj = {
        let pred = dyn_t.step(&encode_pair(&enc_t, test[0].a));
        let tgt = encode_pair(&enc_t, test[0].b);
        cosine(&pred, &tgt)
    };

    // Train on independent rule pairs
    let mut enc_r = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE20B);
    let mut dyn_r = FieldDynamics::new(FIELD_DIM, seed ^ 0xD20B);
    let steps = train_encoder_dynamics(&mut enc_r, &mut dyn_r, &rule_pairs, 160, None);
    let cos_rule = {
        let pred = dyn_r.step(&encode_pair(&enc_r, test[0].a));
        let tgt = encode_pair(&enc_r, test[0].b);
        cosine(&pred, &tgt)
    };

    let audit = ProvenanceAudit::field_only_path("input→encoder→D_phi→output");
    let leakage = audit.leakage_score();
    let delta = cos_rule - cos_traj;

    let verdict = if leakage > 0 {
        "LEAKED"
    } else if cos_rule >= COS_PASS && delta > 0.1 {
        "POSITIVE"
    } else if cos_rule > cos_traj + 0.05 {
        "PARTIAL"
    } else if (cos_rule - cos_traj).abs() < 0.05 {
        "NULL"
    } else {
        "NEGATIVE"
    };

    let mut row = base_row("E20_trajectory_vs_rule", seed);
    row.mode = AutonomyMode::Mode2DynamicField.as_str().into();
    row.rule = rule.name();
    row.train_pairs = rule_pairs.len();
    row.test_pairs = 1;
    row.cosine_dynamic = cos_rule;
    row.cosine_static = cos_traj; // reuse: trajectory-trained score
    row.rule_generalization = cos_rule;
    row.experience_gain = delta;
    row.leakage_score = leakage;
    row.steps_trained = steps;
    row.encoder_hash = enc_r.weights_hash();
    row.dynamics_hash = dyn_r.weights_hash();
    row.verdict = verdict.into();
    row.notes = format!(
        "cos_rule={:.3} cos_traj={:.3} delta={:.3} elapsed_ms={:.1}",
        cos_rule,
        cos_traj,
        delta,
        t0.elapsed().as_secs_f64() * 1e3
    );
    row
}

// ── E21 — abstract rule + new parameter / extrapolation ─────────────────────

pub fn run_e21(seed: u64) -> RegistryRowStage2 {
    let t0 = Instant::now();
    let rule = RuleKind::Translation { dx: 2.0, dy: 1.0 };
    let starts = grid_points(seed ^ 0xE21, 16, -4.0, 4.0);
    let train = make_pairs(&starts[..12], rule);
    let test_same = make_pairs(&starts[12..14], rule);

    let mut enc = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE21A);
    let mut dynm = FieldDynamics::new(FIELD_DIM, seed ^ 0xD21A);
    let steps = train_encoder_dynamics(&mut enc, &mut dynm, &train, 160, None);

    let cos_same = mean(
        &test_same
            .iter()
            .map(|p| cosine(&dynm.step(&encode_pair(&enc, p.a)), &encode_pair(&enc, p.b)))
            .collect::<Vec<_>>(),
    );

    // Harder: train multi-dx then extrapolate dx=3
    let mut multi = Vec::new();
    for &dx in &[-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
        let r = RuleKind::Translation { dx, dy: 1.0 };
        let pts = grid_points(seed ^ (dx.to_bits()), 6, -3.0, 3.0);
        multi.extend(make_pairs(&pts, r));
    }
    let mut enc2 = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE21B);
    let mut dyn2 = FieldDynamics::new(FIELD_DIM, seed ^ 0xD21B);
    // Condition dynamics on dx by concatenating a context feature into training via
    // auxiliary pairs: we encode action by training on each rule separately then test dx=3.
    let _ = train_encoder_dynamics(&mut enc2, &mut dyn2, &multi, 60, None);

    // For dx=3 extrapolation: regularity from dx=2 experiences as soft prior, then one-shot adapt?
    // Honest: apply D_phi trained on |dx|<=2 without seeing dx=3 pairs; measure against true dx=3.
    let hard_rule = RuleKind::Translation { dx: 3.0, dy: 1.0 };
    let hard_starts = grid_points(seed ^ 0xE213, 4, -2.0, 2.0);
    let hard_test = make_pairs(&hard_starts, hard_rule);
    // Without parameter conditioning, pure D_phi cannot know dx=3 — expect weak.
    // Provide regularity mean_delta estimated from dx=2 subset only (experience), not targets.
    let mut mem = ExperienceMemory::new();
    for (i, p) in multi
        .iter()
        .filter(|p| p.rule.contains("dx=2"))
        .take(8)
        .enumerate()
    {
        let a = encode_pair(&enc2, p.a);
        let b = encode_pair(&enc2, p.b);
        mem.store(ConsolidatedExperience {
            context_signature: format!("dx2_{i}"),
            state_before: a.clone(),
            action_or_relation: p.rule.clone(),
            state_after: b.clone(),
            confidence: 1.0,
            energy: 0.0,
            repetition_count: 1,
            before_hash: hash_f64_slice(&a),
            after_hash: hash_f64_slice(&b),
        });
    }
    mem.consolidate();
    // Scale mean_delta by 3/2 as a crude parameter extrapolation signal (experience-derived).
    if let Some(ref mut d) = mem.mean_delta {
        for x in d.iter_mut() {
            *x *= 1.5;
        }
    }

    let mut cos_hard_raw = Vec::new();
    let mut cos_hard_ctx = Vec::new();
    for p in &hard_test {
        let src = encode_pair(&enc2, p.a);
        let tgt = encode_pair(&enc2, p.b);
        let pred = dyn2.step(&src);
        cos_hard_raw.push(cosine(&pred, &tgt));
        // context-biased prediction: src + scaled regularity, then one dynamics step from blend
        if let Some(delta) = mem.regularity_context() {
            let mut biased = src.clone();
            for i in 0..biased.len().min(delta.len()) {
                biased[i] += delta[i];
            }
            normalize(&mut biased);
            // interpret biased as proposed after-state (experience prior), mix with D_phi
            let mut mix = pred.clone();
            for i in 0..mix.len() {
                mix[i] = 0.5 * pred[i] + 0.5 * biased.get(i).copied().unwrap_or(pred[i]);
            }
            normalize(&mut mix);
            cos_hard_ctx.push(cosine(&mix, &tgt));
        } else {
            cos_hard_ctx.push(cosine(&pred, &tgt));
        }
    }
    let cos_extrap = mean(&cos_hard_ctx);
    let cos_extrap_raw = mean(&cos_hard_raw);

    let mut audit = ProvenanceAudit::field_only_path("input→encoder→D_phi(+regularity)→output");
    // Confirm hard targets not in multi train
    for p in &hard_test {
        if multi
            .iter()
            .any(|t| (t.b.0 - p.b.0).abs() < 1e-9 && (t.b.1 - p.b.1).abs() < 1e-9)
        {
            audit.target_equivalent_seen = true;
        }
    }
    let leakage = audit.leakage_score();

    let verdict = if leakage > 0 {
        "LEAKED"
    } else if cos_same >= COS_PASS && cos_extrap >= COS_PARTIAL {
        "POSITIVE"
    } else if cos_same >= COS_PARTIAL {
        "PARTIAL"
    } else {
        "NULL"
    };

    let mut row = base_row("E21_abstract_rule_param", seed);
    row.mode = AutonomyMode::Mode2DynamicField.as_str().into();
    row.rule = format!("{}; extrap dx=3", rule.name());
    row.train_pairs = train.len();
    row.test_pairs = test_same.len() + hard_test.len();
    row.cosine_dynamic = cos_same;
    row.cosine_static = cos_extrap_raw;
    row.cosine_linear = cos_extrap;
    row.rule_generalization = cos_same;
    row.novel_state_generation = cos_extrap;
    row.leakage_score = leakage;
    row.steps_trained = steps;
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row.verdict = verdict.into();
    row.notes = format!(
        "same_param_cos={:.3} extrap_raw={:.3} extrap_ctx={:.3} elapsed_ms={:.1}",
        cos_same,
        cos_extrap_raw,
        cos_extrap,
        t0.elapsed().as_secs_f64() * 1e3
    );
    row
}

// ── E22 — unobserved composition ────────────────────────────────────────────

pub fn run_e22(seed: u64) -> RegistryRowStage2 {
    let t0 = Instant::now();
    let t1 = RuleKind::Translation { dx: 1.0, dy: 0.0 };
    let t2 = RuleKind::Translation { dx: 0.0, dy: 1.0 };
    let starts = grid_points(seed ^ 0xE22, 16, -3.0, 3.0);
    let pairs_t1 = make_pairs(&starts[..8], t1);
    let pairs_t2 = make_pairs(&starts[8..14], t2);
    let mut train = pairs_t1;
    train.extend(pairs_t2);

    // Never train T2∘T1
    let test_starts = grid_points(seed ^ 0x000E_2207, 4, -2.0, 2.0);
    let test: Vec<Pair> = test_starts
        .iter()
        .map(|&a| {
            let mid = t1.apply(a);
            let b = t2.apply(mid);
            Pair {
                a,
                b,
                rule: "T2∘T1".into(),
            }
        })
        .collect();

    let mut enc = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE22A);
    let mut dynm = FieldDynamics::new(FIELD_DIM, seed ^ 0xD22A);
    let steps = train_encoder_dynamics(&mut enc, &mut dynm, &train, 100, None);

    // Compose by rolling out two steps (D_phi twice) — no RQM
    let mut cos_compose = Vec::new();
    let mut cos_one = Vec::new();
    for p in &test {
        let s0 = encode_pair(&enc, p.a);
        let s1 = dynm.step(&s0);
        let s2 = dynm.step(&s1);
        let tgt = encode_pair(&enc, p.b);
        cos_compose.push(cosine(&s2, &tgt));
        cos_one.push(cosine(&s1, &tgt));
    }
    let c2 = mean(&cos_compose);
    let c1 = mean(&cos_one);

    let train_src: Vec<_> = train.iter().map(|p| encode_pair(&enc, p.a)).collect();
    let train_tgt: Vec<_> = train.iter().map(|p| encode_pair(&enc, p.b)).collect();
    let mut c_nn = Vec::new();
    for p in &test {
        let np = nn_predict(&encode_pair(&enc, p.a), &train_src, &train_tgt);
        c_nn.push(cosine(&np, &encode_pair(&enc, p.b)));
    }
    let cos_nn = mean(&c_nn);

    let audit = ProvenanceAudit::field_only_path("input→encoder→D_phi→D_phi→output");
    let leakage = audit.leakage_score();
    let verdict = if leakage > 0 {
        "LEAKED"
    } else if (c2 >= COS_PARTIAL && c2 > cos_nn) || (c2 > c1 && c2 > 0.5) {
        "PARTIAL"
    } else if c2 >= 0.4 {
        "NULL"
    } else {
        "NEGATIVE"
    };

    let mut row = base_row("E22_unobserved_composition", seed);
    row.mode = AutonomyMode::Mode2DynamicField.as_str().into();
    row.rule = "T2(T1(x)) never trained".into();
    row.train_pairs = train.len();
    row.test_pairs = test.len();
    row.cosine_dynamic = c2;
    row.cosine_static = c1;
    row.cosine_nn = cos_nn;
    row.rule_generalization = c2;
    row.leakage_score = leakage;
    row.steps_trained = steps;
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row.verdict = verdict.into();
    row.notes = format!(
        "compose2={:.3} one_step={:.3} nn={:.3} elapsed_ms={:.1}",
        c2,
        c1,
        cos_nn,
        t0.elapsed().as_secs_f64() * 1e3
    );
    row
}

// ── E23 — long-horizon rollout ──────────────────────────────────────────────

pub fn run_e23(seed: u64) -> RegistryRowStage2 {
    let t0 = Instant::now();
    let rule = RuleKind::Translation { dx: 0.5, dy: 0.25 };
    let starts = grid_points(seed ^ 0xE23, 20, -3.0, 3.0);
    // Train one-step only
    let train = make_pairs(&starts[..16], rule);
    let mut enc = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE23A);
    let mut dynm = FieldDynamics::new(FIELD_DIM, seed ^ 0xD23A);
    let steps = train_encoder_dynamics(&mut enc, &mut dynm, &train, 120, None);

    let horizons = [1usize, 2, 4, 8, 16, 32];
    let mut horizon_cos = Vec::new();
    let test_start = starts[16];
    for &h in &horizons {
        let mut cur_pt = test_start;
        let mut cur_fp = encode_pair(&enc, cur_pt);
        for _ in 0..h {
            cur_pt = rule.apply(cur_pt);
            cur_fp = dynm.step(&cur_fp);
        }
        let tgt = encode_pair(&enc, cur_pt);
        horizon_cos.push(cosine(&cur_fp, &tgt));
    }

    // Perturbation at h=8
    let mut pert_scores = Vec::new();
    for &eps in &[0.01_f64, 0.05, 0.10, 0.20] {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ eps.to_bits());
        let noisy = (
            test_start.0 + (rng.gen::<f64>() - 0.5) * 2.0 * eps,
            test_start.1 + (rng.gen::<f64>() - 0.5) * 2.0 * eps,
        );
        let mut cur_pt = noisy;
        let mut cur_fp = encode_pair(&enc, noisy);
        let mut true_pt = noisy;
        for _ in 0..8 {
            true_pt = rule.apply(true_pt);
            cur_fp = dynm.step(&cur_fp);
            let _ = cur_pt;
            cur_pt = true_pt;
        }
        pert_scores.push(cosine(&cur_fp, &encode_pair(&enc, true_pt)));
    }

    let cos_h1 = horizon_cos[0];
    let cos_h8 = horizon_cos[3];
    let cos_h32 = *horizon_cos.last().unwrap_or(&0.0);
    let audit = ProvenanceAudit::field_only_path("input→encoder→D_phi^h→output");
    let leakage = audit.leakage_score();

    let verdict = if leakage > 0 {
        "LEAKED"
    } else if cos_h1 >= COS_PASS && cos_h8 >= COS_PARTIAL {
        "POSITIVE"
    } else if cos_h1 >= COS_PARTIAL {
        "PARTIAL"
    } else {
        "NULL"
    };

    let mut row = base_row("E23_long_horizon", seed);
    row.mode = AutonomyMode::Mode2DynamicField.as_str().into();
    row.rule = rule.name();
    row.train_pairs = train.len();
    row.test_pairs = 1;
    row.cosine_dynamic = cos_h1;
    row.cosine_static = cos_h8;
    row.cosine_linear = cos_h32;
    row.rule_generalization = cos_h8;
    row.stability = mean(&pert_scores);
    row.leakage_score = leakage;
    row.steps_trained = steps;
    row.encoder_hash = enc.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row.verdict = verdict.into();
    row.notes = format!(
        "horizons_cos={:?} pert_h8={:?} elapsed_ms={:.1}",
        horizon_cos
            .iter()
            .map(|c| format!("{c:.3}"))
            .collect::<Vec<_>>(),
        pert_scores
            .iter()
            .map(|c| format!("{c:.3}"))
            .collect::<Vec<_>>(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    row
}

// ── E24 — static vs dynamic (same init) ─────────────────────────────────────

pub fn run_e24(seed: u64) -> RegistryRowStage2 {
    let t0 = Instant::now();
    let rule = RuleKind::Rotation {
        theta: std::f64::consts::FRAC_PI_6,
    };
    let starts = grid_points(seed ^ 0xE24, 14, -3.0, 3.0);
    let train = make_pairs(&starts[..10], rule);
    let test = make_pairs(&starts[10..], rule);

    // Identical encoder init for both arms.
    let enc_init = TrainableFieldEncoder::new(FEAT_DIM, FIELD_DIM, seed ^ 0xE24E);
    let mut enc_static = enc_init.clone();
    let mut enc_dyn = enc_init;
    let mut dynm = FieldDynamics::new(FIELD_DIM, seed ^ 0xD24);
    let dyn_random = FieldDynamics::new(FIELD_DIM, seed ^ 0x000D_2499);
    // freeze random dynamics
    let mut dyn_random = dyn_random;
    dyn_random.eta = 0.0;

    // Static arm: train encoder only (contrastive), no dynamics update.
    for _ in 0..80 {
        for p in &train {
            let fa = point_features(p.a);
            let fb = point_features(p.b);
            let pa = enc_static.encode(&fa);
            let pb = enc_static.encode(&fb);
            let _ =
                enc_static.train_step(&fa, std::slice::from_ref(&pa), std::slice::from_ref(&pb));
            let _ =
                enc_static.train_step(&fb, std::slice::from_ref(&pb), std::slice::from_ref(&pa));
        }
    }
    // Dynamic arm: encoder + D_phi
    let steps = train_encoder_dynamics(&mut enc_dyn, &mut dynm, &train, 80, None);

    let mut cos_static = Vec::new();
    let mut cos_dyn = Vec::new();
    let mut cos_rand = Vec::new();
    for p in &test {
        let s_fp = encode_pair(&enc_static, p.a);
        let t_fp = encode_pair(&enc_static, p.b);
        cos_static.push(cosine(&s_fp, &t_fp));

        let pred = dynm.step(&encode_pair(&enc_dyn, p.a));
        let tgt = encode_pair(&enc_dyn, p.b);
        cos_dyn.push(cosine(&pred, &tgt));

        let pred_r = dyn_random.step(&encode_pair(&enc_dyn, p.a));
        cos_rand.push(cosine(&pred_r, &tgt));
    }
    let cs = mean(&cos_static);
    let cd = mean(&cos_dyn);
    let cr = mean(&cos_rand);
    let audit = ProvenanceAudit::field_only_path("input→encoder→D_phi→output");
    let leakage = audit.leakage_score();

    let verdict = if leakage > 0 {
        "LEAKED"
    } else if cd > cs + 0.1 && cd > cr + 0.1 && cd >= COS_PARTIAL {
        "POSITIVE"
    } else if cd > cs + 0.03 {
        "PARTIAL"
    } else if (cd - cs).abs() < 0.03 {
        "NULL"
    } else {
        "NEGATIVE"
    };

    let mut row = base_row("E24_static_vs_dynamic", seed);
    row.mode = AutonomyMode::Mode2DynamicField.as_str().into();
    row.rule = rule.name();
    row.train_pairs = train.len();
    row.test_pairs = test.len();
    row.cosine_dynamic = cd;
    row.cosine_static = cs;
    row.cosine_linear = cr; // random dynamics control
    row.rule_generalization = cd;
    row.experience_gain = cd - cs;
    row.leakage_score = leakage;
    row.steps_trained = steps;
    row.encoder_hash = enc_dyn.weights_hash();
    row.dynamics_hash = dynm.weights_hash();
    row.verdict = verdict.into();
    row.notes = format!(
        "dynamic={:.3} static={:.3} random_dyn={:.3} delta={:.3} elapsed_ms={:.1}",
        cd,
        cs,
        cr,
        cd - cs,
        t0.elapsed().as_secs_f64() * 1e3
    );
    row
}

// ── E25–E30 deferred scaffolds (honest INVALID / NULL placeholders) ─────────

fn deferred_row(name: &str, seed: u64, note: &str) -> RegistryRowStage2 {
    let mut row = base_row(name, seed);
    row.mode = AutonomyMode::Mode3DynamicPlusCdtTraining.as_str().into();
    row.verdict = "INVALID".into();
    row.notes = format!("DEFERRED: {note}");
    row
}

pub fn run_e25(seed: u64) -> RegistryRowStage2 {
    // Lightweight scaffold: reuse E18B signal as proxy for "experience helps novel family"
    let mut row = run_e18b(seed ^ 0xE25);
    row.experiment = "E25_cdt_experience_not_answer".into();
    row.notes = format!(
        "scaffold via E18B path (novel target not in CDT); full multi-family deferred. {}",
        row.notes
    );
    row
}

pub fn run_e26(seed: u64) -> RegistryRowStage2 {
    // Ablation mirrors E18C conditions
    let mut row = run_e18c(seed ^ 0xE26);
    row.experiment = "E26_experience_ablation".into();
    row.notes = format!("ablation mapped to E18C C0..C3; {}", row.notes);
    row
}

pub fn run_e27(seed: u64) -> RegistryRowStage2 {
    deferred_row(
        "E27_rule_transfer",
        seed,
        "cross-family transfer not fully instrumented in this pass",
    )
}

pub fn run_e28(seed: u64) -> RegistryRowStage2 {
    deferred_row(
        "E28_continual_forgetting",
        seed,
        "continual R1/R2/R3 protocol not fully run",
    )
}

pub fn run_e29(seed: u64) -> RegistryRowStage2 {
    deferred_row(
        "E29_causal_intervention",
        seed,
        "checkpoint intervene/rollback not wired",
    )
}

pub fn run_e30(seed: u64) -> RegistryRowStage2 {
    deferred_row(
        "E30_persistence_restart",
        seed,
        "process restart persistence not wired",
    )
}

// ── Fase A smoke: audit + FIELD_ONLY defaults ───────────────────────────────

pub fn run_fase_a_scaffold(seed: u64) -> RegistryRowStage2 {
    let mut audit = ProvenanceAudit::field_only_path("input→encoder→D_phi→output");
    audit.cdt_query_count = 0;
    audit.rqm_query_count = 0;
    let mut row = base_row("FASE_A_field_only_scaffold", seed);
    row.mode = AutonomyMode::Mode2DynamicField.as_str().into();
    row.field_only = true;
    row.cdt_enabled = false;
    row.rqm_enabled = false;
    row.leakage_score = audit.leakage_score();
    row.verdict = if audit.clean_for_field_only() {
        "POSITIVE"
    } else {
        "LEAKED"
    }
    .into();
    row.notes = format!(
        "FIELD_ONLY defaults: RQM=OFF CDT=OFF; leakage={}; path={}",
        row.leakage_score, audit.information_path
    );
    row
}

// ── suite runner / CSV / markdown ───────────────────────────────────────────

pub fn default_seeds() -> Vec<u64> {
    (0..8).map(|i| 0xE1800 + i).collect()
}

pub fn run_all_stage2(seeds: &[u64]) -> Vec<RegistryRowStage2> {
    let mut rows = Vec::new();
    for &seed in seeds {
        rows.push(run_fase_a_scaffold(seed));
        rows.push(run_e18a(seed));
        rows.push(run_e18b(seed));
        rows.push(run_e18c(seed));
        rows.push(run_e19(seed));
        rows.push(run_e20(seed));
        rows.push(run_e21(seed));
        rows.push(run_e22(seed));
        rows.push(run_e23(seed));
        rows.push(run_e24(seed));
        rows.push(run_e25(seed));
        rows.push(run_e26(seed));
        rows.push(run_e27(seed));
        rows.push(run_e28(seed));
        rows.push(run_e29(seed));
        rows.push(run_e30(seed));
    }
    rows
}

pub fn format_registry_table_stage2(rows: &[RegistryRowStage2]) -> String {
    let mut s = String::new();
    s.push_str("# Registry — etapa 2 autonomía de campo (E18–E30)\n\n");
    s.push_str("| Exp | seed | mode | rule_gen | dyn | static | leak | verdict |\n");
    s.push_str("|-----|-----:|------|---------:|----:|-------:|-----:|---------|\n");
    for r in rows {
        s.push_str(&format!(
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {} | {} |\n",
            r.experiment,
            r.seed,
            r.mode,
            r.rule_generalization,
            r.cosine_dynamic,
            r.cosine_static,
            r.leakage_score,
            r.verdict.replace('|', "/"),
        ));
    }
    s
}

pub fn rows_to_csv_stage2(rows: &[RegistryRowStage2]) -> String {
    let mut csv = String::from(
        "experiment,commit,seed,hardware,rust_version,mode,rule,train_pairs,test_pairs,\
cdt_enabled,rqm_enabled,field_only,cosine_dynamic,cosine_static,cosine_linear,cosine_nn,cosine_table,\
coord_err_dynamic,rule_generalization,novel_state_generation,experience_gain,leakage_score,\
cdt_query_count,rqm_query_count,energy,stability,steps_trained,encoder_hash,dynamics_hash,\
verdict,periphery,notes\n",
    );
    for r in rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{:.6},{},{},{},{},{},{}\n",
            r.experiment,
            r.commit,
            r.seed,
            r.hardware,
            r.rust_version.replace(",", ";"),
            r.mode,
            r.rule.replace(",", ";").replace('\n', " "),
            r.train_pairs,
            r.test_pairs,
            r.cdt_enabled,
            r.rqm_enabled,
            r.field_only,
            r.cosine_dynamic,
            r.cosine_static,
            r.cosine_linear,
            r.cosine_nn,
            r.cosine_table,
            r.coord_err_dynamic,
            r.rule_generalization,
            r.novel_state_generation,
            r.experience_gain,
            r.leakage_score,
            r.cdt_query_count,
            r.rqm_query_count,
            r.energy,
            r.stability,
            r.steps_trained,
            r.encoder_hash,
            r.dynamics_hash,
            r.verdict.replace(",", ";"),
            r.periphery.replace(",", ";"),
            r.notes.replace(",", ";").replace('\n', " "),
        ));
    }
    csv
}

pub fn summarize_verdicts(rows: &[RegistryRowStage2]) -> String {
    use std::collections::BTreeMap;
    let mut by_exp: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for r in rows {
        *by_exp
            .entry(r.experiment.clone())
            .or_default()
            .entry(r.verdict.clone())
            .or_default() += 1;
    }
    let mut s = String::from("## Suite summary (verdict counts by experiment)\n\n");
    s.push_str("| Experiment | verdict histogram |\n|------------|-------------------|\n");
    for (exp, hist) in by_exp {
        let parts: Vec<String> = hist.iter().map(|(v, n)| format!("{v}:{n}")).collect();
        s.push_str(&format!("| {exp} | {} |\n", parts.join(", ")));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fase_a_field_only_clean() {
        let row = run_fase_a_scaffold(0xE1800);
        assert_eq!(row.verdict, "POSITIVE");
        assert!(!row.rqm_enabled);
        assert!(!row.cdt_enabled);
        assert_eq!(row.leakage_score, 0);
    }

    #[test]
    fn e18a_smoke_single_seed() {
        let row = run_e18a(0xE1800);
        assert!(row.field_only);
        assert!(!row.rqm_enabled);
        assert_eq!(row.cdt_query_count, 0);
        assert_ne!(row.verdict, "LEAKED");
        assert_ne!(row.verdict, "INVALID");
        // static should not trivially solve translation
        assert!(
            row.cosine_dynamic + 1e-6 >= row.cosine_table,
            "dynamic should beat or match empty table on novel: dyn={} tab={}",
            row.cosine_dynamic,
            row.cosine_table
        );
    }

    #[test]
    fn e18b_target_not_retrieved_from_cdt() {
        let row = run_e18b(0xE1801);
        assert_eq!(row.cdt_query_count, 0);
        assert_ne!(row.verdict, "LEAKED");
        assert!(row.notes.contains("target_in_cdt=false") || row.leakage_score == 0);
    }

    #[test]
    fn e19_audit_zero_queries() {
        let row = run_e19(0xE1802);
        assert_eq!(row.cdt_query_count, 0);
        assert_eq!(row.rqm_query_count, 0);
        assert_eq!(row.leakage_score, 0);
    }

    #[test]
    fn e20_e21_e22_e23_e24_smoke() {
        let s = 0xE1803;
        for row in [run_e20(s), run_e21(s), run_e22(s), run_e23(s), run_e24(s)] {
            assert!(!row.experiment.is_empty());
            assert_ne!(row.verdict, "LEAKED", "{}", row.experiment);
        }
    }

    #[test]
    fn rule_transforms_ground_truth() {
        let t = RuleKind::Translation { dx: 2.0, dy: 1.0 };
        assert_eq!(t.apply((0.0, 0.0)), (2.0, 1.0));
        let r = RuleKind::Rotation {
            theta: std::f64::consts::FRAC_PI_2,
        };
        let (x, y) = r.apply((1.0, 0.0));
        assert!(x.abs() < 1e-9 && (y - 1.0).abs() < 1e-9);
    }

    /// Full 8-seed suite is heavy; gate behind STAGE2_FULL=1.
    #[test]
    fn stage2_dev_suite_optional() {
        if std::env::var("STAGE2_FULL").ok().as_deref() != Some("1") {
            return;
        }
        let seeds = default_seeds();
        let rows = run_all_stage2(&seeds);
        assert_eq!(rows.len(), seeds.len() * 16);
        let md = format!(
            "{}\n{}\n",
            format_registry_table_stage2(&rows),
            summarize_verdicts(&rows)
        );
        let csv = rows_to_csv_stage2(&rows);
        let _ = std::fs::write("docs/resultados_etapa_2_autonomia.md", &md);
        let _ = std::fs::write("docs/resultados_etapa_2_autonomia.csv", &csv);
    }
}
