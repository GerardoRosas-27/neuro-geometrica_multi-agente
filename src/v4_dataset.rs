//! v4 dataset generator (plan v4 §5, protocolo §2 `v4_dataset`).
//!
//! Multi-family geometric rule suite. Each example is an *object* of `k`
//! 2-D points transformed by a parameterised rule. TRAIN/DEV/TEST are
//! generated from independent RNG streams, TEST is sealed (SHA-256) before
//! any training, and an anti-contamination audit checks exact and
//! canonical-equivalent targets (permutation-invariant, quantised).
//!
//! Nothing here is reused from E11–E17/v3: fresh generator, fresh seeds.

#![allow(clippy::needless_range_loop)]

use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt::Write as _;

pub const V4_GENERATOR_VERSION: &str = "v4_gen_1.0.0";

/// Rule families (plan v4 §5: translation, rotation, reflection, scale,
/// shear, affine, composition and a smooth non-linear family).
pub const FAMILIES: [&str; 8] = [
    "translate",
    "rotate",
    "reflect",
    "scale",
    "shear",
    "affine",
    "compose_rot_scale",
    "radial_warp",
];
pub const N_FAM: usize = FAMILIES.len();
/// Context = one-hot family + 2 normalised parameters.
pub const CTX_DIM: usize = N_FAM + 2;

/// Interpolation hole on p1: TRAIN/DEV never sample p1 in this open band.
pub const HOLE: (f64, f64) = (0.10, 0.45);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Partition {
    Train,
    Dev,
    Test,
}

impl Partition {
    pub fn as_str(self) -> &'static str {
        match self {
            Partition::Train => "TRAIN",
            Partition::Dev => "DEV",
            Partition::Test => "TEST",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectClass {
    Blob,
    Circle,
    Triangle,
}

impl ObjectClass {
    pub fn as_str(self) -> &'static str {
        match self {
            ObjectClass::Blob => "blob",
            ObjectClass::Circle => "circle",
            ObjectClass::Triangle => "triangle",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Example {
    pub rule_id: usize,
    pub family_id: usize,
    pub instance_id: u64,
    pub object_class: ObjectClass,
    pub params: [f64; 2],
    pub input: Vec<f64>,
    pub context: Vec<f64>,
    pub expected: Vec<f64>,
    pub canonical_target_hash: u64,
    pub partition: Partition,
    /// "in_dist" or "interp_hole" (TEST only).
    pub subset: &'static str,
}

/// Apply one rule to one 2-D point.
pub fn apply_point(fam: usize, p: [f64; 2], q: (f64, f64)) -> (f64, f64) {
    let (x, y) = q;
    match fam {
        0 => (x + 0.5 * p[0], y + 0.5 * p[1]),
        1 => {
            let t = 0.6 * p[0] + 0.15 * p[1];
            let (c, s) = (t.cos(), t.sin());
            (c * x - s * y, s * x + c * y)
        }
        2 => {
            let a = 0.8 * p[0] + 0.2 * p[1];
            let (c, s) = ((2.0 * a).cos(), (2.0 * a).sin());
            (c * x + s * y, s * x - c * y)
        }
        3 => ((0.3 * p[0]).exp() * x, (0.3 * p[1]).exp() * y),
        4 => (x + 0.5 * p[0] * y, y + 0.3 * p[1] * x),
        5 => (
            x + 0.3 * (p[0] * x + p[1] * y) + 0.3 * p[1],
            y + 0.3 * (-p[1] * x + p[0] * y) + 0.3 * p[0],
        ),
        6 => {
            let t = 0.6 * p[0];
            let sc = (0.3 * p[1]).exp();
            let (c, s) = (t.cos(), t.sin());
            (sc * (c * x - s * y), sc * (s * x + c * y))
        }
        _ => {
            let r2 = x * x + y * y;
            let g = 1.0 + 0.3 * p[0] * (r2 - 1.0).tanh();
            (g * x + 0.2 * p[1] * y.sin(), g * y + 0.2 * p[1] * x.sin())
        }
    }
}

pub fn apply_rule(fam: usize, p: [f64; 2], obj: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; obj.len()];
    for i in 0..obj.len() / 2 {
        let (a, b) = apply_point(fam, p, (obj[2 * i], obj[2 * i + 1]));
        out[2 * i] = a;
        out[2 * i + 1] = b;
    }
    out
}

pub fn context_of(fam: usize, p: [f64; 2]) -> Vec<f64> {
    let mut c = vec![0.0; CTX_DIM];
    c[fam] = 1.0;
    c[N_FAM] = p[0];
    c[N_FAM + 1] = p[1];
    c
}

pub fn make_object(rng: &mut Xoshiro256StarStar, k: usize, class: ObjectClass) -> Vec<f64> {
    let cx = rng.gen_range(-0.5..0.5);
    let cy = rng.gen_range(-0.5..0.5);
    let r = rng.gen_range(0.5..1.0);
    let phase = rng.gen_range(0.0..std::f64::consts::TAU);
    let mut v = vec![0.0; 2 * k];
    for i in 0..k {
        let (px, py) = match class {
            ObjectClass::Blob => {
                let a = phase
                    + std::f64::consts::TAU * (i as f64 + rng.gen_range(-0.3..0.3)) / k as f64;
                let rr = r * rng.gen_range(0.6..1.2);
                (rr * a.cos(), rr * a.sin())
            }
            ObjectClass::Circle => {
                let a = phase + std::f64::consts::TAU * i as f64 / k as f64;
                (r * a.cos(), r * a.sin())
            }
            ObjectClass::Triangle => {
                let t = 3.0 * i as f64 / k as f64;
                let e = t.floor();
                let f = t - e;
                let a0 = phase + std::f64::consts::TAU * e / 3.0;
                let a1 = phase + std::f64::consts::TAU * (e + 1.0) / 3.0;
                (
                    r * ((1.0 - f) * a0.cos() + f * a1.cos()),
                    r * ((1.0 - f) * a0.sin() + f * a1.sin()),
                )
            }
        };
        v[2 * i] = cx + px;
        v[2 * i + 1] = cy + py;
    }
    v
}

/// Permutation-invariant quantised hash (canonical equivalence: the same
/// point set in any order / any family producing it collides).
pub fn canonical_hash(v: &[f64]) -> u64 {
    let mut pts: Vec<(i64, i64)> = (0..v.len() / 2)
        .map(|i| {
            (
                (v[2 * i] * 1000.0).round() as i64,
                (v[2 * i + 1] * 1000.0).round() as i64,
            )
        })
        .collect();
    pts.sort_unstable();
    let mut h: u64 = 0xcbf29ce484222325;
    for (a, b) in pts {
        for x in [a, b] {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Sorted point list for near-duplicate L2 checks.
pub fn canonical_points(v: &[f64]) -> Vec<f64> {
    let mut pts: Vec<(f64, f64)> = (0..v.len() / 2).map(|i| (v[2 * i], v[2 * i + 1])).collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.into_iter().flat_map(|(a, b)| [a, b]).collect()
}

#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub k_points: usize,
    pub families: Vec<usize>,
    pub train_per_family: usize,
    pub dev_per_family: usize,
    pub test_per_family: usize,
    pub train_class: ObjectClass,
    pub test_class: ObjectClass,
}

impl DatasetConfig {
    pub fn standard(k: usize) -> Self {
        Self {
            k_points: k,
            families: (0..N_FAM).collect(),
            train_per_family: 24,
            dev_per_family: 12,
            test_per_family: 32,
            train_class: ObjectClass::Blob,
            test_class: ObjectClass::Blob,
        }
    }

    pub fn hash(&self) -> String {
        sha256_hex(format!("{self:?}").as_bytes())
    }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub seed: u64,
    pub config: DatasetConfig,
    pub train: Vec<Example>,
    pub dev: Vec<Example>,
    pub test: Vec<Example>,
}

fn sample_params(rng: &mut Xoshiro256StarStar, in_hole: bool) -> [f64; 2] {
    let p1 = if in_hole {
        rng.gen_range((HOLE.0 + 0.01)..(HOLE.1 - 0.01))
    } else {
        loop {
            let v: f64 = rng.gen_range(-1.0..1.0);
            if v <= HOLE.0 || v >= HOLE.1 {
                break v;
            }
        }
    };
    [p1, rng.gen_range(-1.0..1.0)]
}

fn gen_partition(
    seed: u64,
    salt: u64,
    cfg: &DatasetConfig,
    n_per: usize,
    part: Partition,
    class: ObjectClass,
) -> Vec<Example> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ salt);
    let mut out = Vec::new();
    for &fam in &cfg.families {
        for i in 0..n_per {
            let hole = part == Partition::Test && i % 2 == 1;
            let p = sample_params(&mut rng, hole);
            let obj = make_object(&mut rng, cfg.k_points, class);
            let y = apply_rule(fam, p, &obj);
            out.push(Example {
                rule_id: fam,
                family_id: fam,
                instance_id: (salt << 32) | ((fam as u64) << 16) | i as u64,
                object_class: class,
                params: p,
                context: context_of(fam, p),
                canonical_target_hash: canonical_hash(&y),
                input: obj,
                expected: y,
                partition: part,
                subset: if hole { "interp_hole" } else { "in_dist" },
            });
        }
    }
    out
}

pub fn generate(seed: u64, cfg: &DatasetConfig) -> Dataset {
    Dataset {
        seed,
        config: cfg.clone(),
        train: gen_partition(
            seed,
            0x7121,
            cfg,
            cfg.train_per_family,
            Partition::Train,
            cfg.train_class,
        ),
        dev: gen_partition(
            seed,
            0xDE77,
            cfg,
            cfg.dev_per_family,
            Partition::Dev,
            cfg.train_class,
        ),
        test: gen_partition(
            seed,
            0x7E57,
            cfg,
            cfg.test_per_family,
            Partition::Test,
            cfg.test_class,
        ),
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let mut s = String::with_capacity(64);
    for b in h.finalize() {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

pub fn serialize(xs: &[Example]) -> Vec<u8> {
    let mut buf = Vec::new();
    for e in xs {
        buf.extend_from_slice(&(e.family_id as u64).to_le_bytes());
        buf.extend_from_slice(&e.instance_id.to_le_bytes());
        for v in e.params.iter().chain(&e.input).chain(&e.expected) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

#[derive(Clone, Debug)]
pub struct Manifest {
    pub generator_version: &'static str,
    pub seed: u64,
    pub config_hash: String,
    pub train_sha256: String,
    pub dev_sha256: String,
    pub test_sha256: String,
    pub n_train: usize,
    pub n_dev: usize,
    pub n_test: usize,
    pub audit_status: &'static str,
    pub exact_overlap: usize,
    pub canonical_overlap: usize,
    pub near_dup: usize,
}

impl Manifest {
    pub fn manifest_hash(&self) -> String {
        sha256_hex(self.to_json().as_bytes())
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"generator_version\":\"{}\",\"seed\":\"0x{:X}\",\"config_hash\":\"{}\",\"train_sha256\":\"{}\",\"dev_sha256\":\"{}\",\"test_sha256\":\"{}\",\"n_train\":{},\"n_dev\":{},\"n_test\":{},\"audit_status\":\"{}\",\"exact_overlap\":{},\"canonical_overlap\":{},\"near_dup\":{}}}",
            self.generator_version,
            self.seed,
            self.config_hash,
            self.train_sha256,
            self.dev_sha256,
            self.test_sha256,
            self.n_train,
            self.n_dev,
            self.n_test,
            self.audit_status,
            self.exact_overlap,
            self.canonical_overlap,
            self.near_dup
        )
    }
}

/// Sealed dataset: TEST hash fixed before any model sees data.
#[derive(Clone, Debug)]
pub struct Sealed {
    pub ds: Dataset,
    pub manifest: Manifest,
    pub test_canonical: HashSet<u64>,
}

impl Sealed {
    /// Must be called right before evaluating TEST: verifies immutability.
    pub fn verify_test(&self) -> Result<(), String> {
        let now = sha256_hex(&serialize(&self.ds.test));
        if now != self.manifest.test_sha256 {
            return Err(format!(
                "TEST mutated: {} != {}",
                now, self.manifest.test_sha256
            ));
        }
        Ok(())
    }

    /// Is `target` exactly or canonically equivalent to a TEST target, or
    /// within L2 `tol` of one (sorted points)?
    pub fn equivalent_to_test(&self, target: &[f64], tol: f64) -> bool {
        if self.test_canonical.contains(&canonical_hash(target)) {
            return true;
        }
        let ct = canonical_points(target);
        self.ds
            .test
            .iter()
            .any(|e| e.expected.len() == ct.len() && l2(&canonical_points(&e.expected), &ct) < tol)
    }
}

pub fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

pub const NEAR_DUP_TOL: f64 = 1e-3;
/// Correctness-ball near-duplicate: a stored target within 5 % relative L2
/// of a TEST target would be scored correct, so it counts as equivalent.
pub const NEAR_DUP_REL: f64 = 0.05;

pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn generate_and_seal(seed: u64, cfg: &DatasetConfig) -> Sealed {
    let ds = generate(seed, cfg);
    let test_canonical: HashSet<u64> = ds.test.iter().map(|e| e.canonical_target_hash).collect();
    let seen: Vec<&Example> = ds.train.iter().chain(&ds.dev).collect();
    let test_exact: HashSet<Vec<u64>> = ds
        .test
        .iter()
        .map(|e| e.expected.iter().map(|v| v.to_bits()).collect())
        .collect();
    let exact_overlap = seen
        .iter()
        .filter(|e| {
            test_exact.contains(&e.expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        })
        .count();
    let canonical_overlap = seen
        .iter()
        .filter(|e| test_canonical.contains(&e.canonical_target_hash))
        .count();
    let test_cp: Vec<Vec<f64>> = ds
        .test
        .iter()
        .map(|e| canonical_points(&e.expected))
        .collect();
    let near_dup = seen
        .iter()
        .filter(|e| {
            let c = canonical_points(&e.expected);
            test_cp
                .iter()
                .any(|t| l2(t, &c) < NEAR_DUP_TOL.max(NEAR_DUP_REL * norm(t)))
        })
        .count();
    let audit_status = if exact_overlap + canonical_overlap + near_dup == 0 {
        "CLEAN"
    } else {
        "DATASET_INVALID"
    };
    let manifest = Manifest {
        generator_version: V4_GENERATOR_VERSION,
        seed,
        config_hash: cfg.hash(),
        train_sha256: sha256_hex(&serialize(&ds.train)),
        dev_sha256: sha256_hex(&serialize(&ds.dev)),
        test_sha256: sha256_hex(&serialize(&ds.test)),
        n_train: ds.train.len(),
        n_dev: ds.dev.len(),
        n_test: ds.test.len(),
        audit_status,
        exact_overlap,
        canonical_overlap,
        near_dup,
    };
    Sealed {
        ds,
        manifest,
        test_canonical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_dataset_is_deterministic_and_clean() {
        let cfg = DatasetConfig::standard(6);
        let a = generate_and_seal(0x5A00, &cfg);
        let b = generate_and_seal(0x5A00, &cfg);
        assert_eq!(a.manifest.test_sha256, b.manifest.test_sha256);
        assert_eq!(a.manifest.audit_status, "CLEAN");
        assert!(a.verify_test().is_ok());
        assert!(a
            .ds
            .train
            .iter()
            .all(|e| e.params[0] <= HOLE.0 || e.params[0] >= HOLE.1));
    }

    #[test]
    fn v4_canonical_hash_is_permutation_invariant() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        let w = vec![0.3, 0.4, 0.1, 0.2];
        assert_eq!(canonical_hash(&v), canonical_hash(&w));
    }
}
