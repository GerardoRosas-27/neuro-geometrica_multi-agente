//! Deterministic intra-arm training checkpoints (infra only; v5 design, reused by v6).
//!
//! Timer: wall-clock every `V6_CKPT_SECS` (fallback `V5_CKPT_SECS`, default
//! 300 s) or, if `V6_CKPT_STEPS` is set, every N optimizer steps (fully
//! deterministic save points). The saved state is the same in both modes.
//!
//! Saves the complete loop state of `Dphi::train_*_ck` (next step, RNG,
//! Adam m/v/t, flat params, running loss/max-grad, the query-counter delta
//! accumulated inside the loop, and caller "extra" state) as exact f64 bits,
//! so a resumed run is bit-identical to an uninterrupted one. Does not change
//! any constant, sampler, schedule or metric.

use crate::v4_provenance::{count_query, snapshot, QueryCounters, Store};
use rand_xoshiro::Xoshiro256StarStar;
use serde_json::{json, Value};
use std::cell::Cell;
use std::path::PathBuf;
use std::time::Instant;

pub struct TrainCkpt<'a> {
    pub path: PathBuf,
    pub every_secs: f64,
    /// Step-based timer (deterministic). Overrides `every_secs` when set.
    pub every_steps: Option<usize>,
    /// Identifies the run config; a file with a different key is ignored.
    pub key: String,
    pub extra_save: &'a dyn Fn() -> Value,
    pub extra_load: &'a dyn Fn(&Value),
    pub last: Cell<Option<Instant>>,
}

impl<'a> TrainCkpt<'a> {
    pub fn new(
        path: PathBuf,
        key: String,
        extra_save: &'a dyn Fn() -> Value,
        extra_load: &'a dyn Fn(&Value),
    ) -> Self {
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
        let every_secs = env("V6_CKPT_SECS").or(env("V5_CKPT_SECS")).unwrap_or(300.0);
        let every_steps = std::env::var("V6_CKPT_STEPS").ok().and_then(|v| v.parse().ok());
        Self {
            path,
            every_secs,
            every_steps,
            key,
            extra_save,
            extra_load,
            last: Cell::new(None),
        }
    }
    /// `next_step` = number of optimizer steps completed.
    pub fn due_at(&self, next_step: usize) -> bool {
        match self.every_steps {
            Some(n) if n > 0 => next_step % n == 0,
            _ => self.due(),
        }
    }
    pub fn due(&self) -> bool {
        match self.last.get() {
            None => {
                self.last.set(Some(Instant::now()));
                false
            }
            Some(t) => t.elapsed().as_secs_f64() >= self.every_secs,
        }
    }
}

pub fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}
pub fn unbits(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|b| f64::from_bits(b.as_u64().unwrap()))
                .collect()
        })
        .unwrap_or_default()
}

fn ctr_json(c: &QueryCounters) -> Value {
    json!([
        c.cdt,
        c.rqm,
        c.table,
        c.nn,
        c.attractor,
        c.direct_memory,
        c.decoder_lookup
    ])
}
fn ctr_restore(v: &Value) {
    let st = [
        Store::Cdt,
        Store::Rqm,
        Store::Table,
        Store::Nn,
        Store::Attractor,
        Store::DirectMemory,
        Store::DecoderLookup,
    ];
    for (i, s) in st.into_iter().enumerate() {
        let n = v[i].as_u64().unwrap_or(0);
        if n > 0 {
            count_query(s, n);
        }
    }
}

/// Loop state shared by `train_onestep_ck` / `train_horizon_ck`.
pub struct LoopState {
    pub next_step: usize,
    pub rng: Xoshiro256StarStar,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub t: i32,
    pub p: Vec<f64>,
    pub last: f64,
    pub max_gn: f64,
}

pub fn save(ck: &TrainCkpt, st: &LoopState, base: &QueryCounters) {
    let v = json!({
        "key": ck.key, "next_step": st.next_step,
        "rng": serde_json::to_value(&st.rng).unwrap(),
        "m": bits(&st.m), "v": bits(&st.v), "t": st.t, "p": bits(&st.p),
        "last": st.last.to_bits(), "max_gn": st.max_gn.to_bits(),
        "ctr": ctr_json(&snapshot().delta(base)),
        "extra": (ck.extra_save)(),
    });
    if let Some(d) = ck.path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let tmp = ck.path.with_extension("tmp");
    if let Ok(mut f) = std::fs::File::create(&tmp) {
        use std::io::Write;
        if f.write_all(v.to_string().as_bytes()).is_ok() && f.sync_all().is_ok() {
            let _ = std::fs::rename(&tmp, &ck.path);
        }
    }
    ck.last.set(Some(Instant::now()));
}

/// Returns the saved state if present and matching; restores counters/extra.
pub fn load(ck: &TrainCkpt) -> Option<LoopState> {
    let t = std::fs::read_to_string(&ck.path).ok()?;
    let v: Value = serde_json::from_str(&t).ok()?;
    if v["key"].as_str()? != ck.key {
        return None;
    }
    let st = LoopState {
        next_step: v["next_step"].as_u64()? as usize,
        rng: serde_json::from_value(v["rng"].clone()).ok()?,
        m: unbits(&v["m"]),
        v: unbits(&v["v"]),
        t: v["t"].as_i64()? as i32,
        p: unbits(&v["p"]),
        last: f64::from_bits(v["last"].as_u64()?),
        max_gn: f64::from_bits(v["max_gn"].as_u64()?),
    };
    ctr_restore(&v["ctr"]);
    (ck.extra_load)(&v["extra"]);
    eprintln!("resume {} at step {}", ck.path.display(), st.next_step);
    Some(st)
}
