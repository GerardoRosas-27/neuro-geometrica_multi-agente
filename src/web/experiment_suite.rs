//! Suite rápida de experimentos para la pestaña Pruebas (UI).
//!
//! Corre smokes ligeros de E8–E10, E13/E15 y Clean-Room v2 (1 seed DEV).
//! **No** usa confirmation seeds 0xB300 ni DEV completo 8 seeds.
//! Independiente del estado del modelo; field_eval se reporta aparte.

/// Suites documentadas para `/api/tests/start` (`suite` query/body).
/// Default UI = [`TestSuiteKind::Smoke`] (fuse identidad/latencia/recall, <~30–60s).
/// `stage2_v2_dev` **no** es el default del botón Pruebas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestSuiteKind {
    /// Solo field_eval del fuse (identidad, latencia líquido, recall). Rápido.
    Smoke,
    /// Smokes E8–E10 + E13/E15 + Clean-Room v2 (1 seed DEV). Decenas de segundos.
    ExperimentsSmoke,
    /// Clean-Room DEV 8 seeds `0xA300–0xA307` (bench; no UI default).
    Stage2V2Dev,
}

impl TestSuiteKind {
    pub const DEFAULT: Self = Self::Smoke;

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::ExperimentsSmoke => "experiments_smoke",
            Self::Stage2V2Dev => "stage2_v2_dev",
        }
    }

    pub fn mode(self) -> &'static str {
        match self {
            Self::Smoke | Self::ExperimentsSmoke => "smoke",
            Self::Stage2V2Dev => "dev",
        }
    }

    pub fn seed_family(self) -> &'static str {
        match self {
            Self::Smoke => "0x51D0_0001",
            Self::ExperimentsSmoke => "ui_smoke+0xA300",
            Self::Stage2V2Dev => "0xA300–0xA307",
        }
    }

    /// `rqm_eval` flag for anti-leak telemetry (autonomy arms = off).
    pub fn rqm_eval(self) -> &'static str {
        match self {
            Self::Smoke => "product_fuse",
            Self::ExperimentsSmoke => "mixed",
            Self::Stage2V2Dev => "off",
        }
    }

    pub fn field_only(self) -> bool {
        matches!(self, Self::Stage2V2Dev)
    }
}

/// Parse body/query `suite`. Unknown → smoke (safe default). Empty/None → smoke.
pub fn parse_test_suite(raw: Option<&str>) -> TestSuiteKind {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("smoke") => TestSuiteKind::Smoke,
        Some("experiments_smoke") | Some("experiments") | Some("ui") => {
            TestSuiteKind::ExperimentsSmoke
        }
        Some("stage2_v2_dev") | Some("stage2_dev") | Some("dev") => TestSuiteKind::Stage2V2Dev,
        // Never treat confirmation as a runnable suite from API.
        Some("confirm") | Some("confirmation") | Some("stage2_v2_confirm") => TestSuiteKind::Smoke,
        Some(_) => TestSuiteKind::Smoke,
    }
}

use crate::field_autonomy_stage2_v2::{run_smoke, DEV_SEEDS};
use crate::liquid_experiments_11_17::{run_experiment_13, run_experiment_15};
use crate::liquid_experiments_8_9_10::{run_experiment_10, run_experiment_8, run_experiment_9};
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;

/// Seed fijo corto para smokes líquidos (no confirmation).
const LIQUID_SMOKE_SEED: u64 = 0x00E8_0010;
/// Seed fijo para E13/E15 smoke UI.
const FIELD_SMOKE_SEED: u64 = 0xE1_100;

#[derive(Clone, Debug, Serialize)]
pub struct SuiteRow {
    pub suite: String,
    pub experiment: String,
    pub verdict: String,
    pub notes: String,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ExperimentSuiteReport {
    pub rows: Vec<SuiteRow>,
    pub verdict_counts: HashMap<String, u64>,
    pub elapsed_ms: f64,
    pub notes: Vec<String>,
}

impl ExperimentSuiteReport {
    pub fn summarize_counts(&self) -> String {
        let mut parts: Vec<String> = self
            .verdict_counts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        parts.sort();
        parts.join(" ")
    }
}

fn bump(counts: &mut HashMap<String, u64>, verdict: &str) {
    *counts.entry(verdict.to_string()).or_insert(0) += 1;
}

/// Batería UI: segundos–decenas de segundos. Sin 0xB300, sin 8 seeds DEV.
pub fn run_ui_experiment_suite() -> ExperimentSuiteReport {
    let t0 = Instant::now();
    let mut rows = Vec::new();
    let mut counts: HashMap<String, u64> = HashMap::new();
    let mut notes = vec![
        "suite UI: smokes rápidos E8–E10 + E13/E15 + Clean-Room v2 (1 seed DEV)".into(),
        "confirmation seeds 0xB300 y DEV 8 seeds diferidos (no UI)".into(),
    ];

    // ── E8–E10 líquido ──────────────────────────────────────────────────────
    {
        let t = Instant::now();
        let (row, _) = run_experiment_8(LIQUID_SMOKE_SEED);
        let elapsed = t.elapsed().as_secs_f64() * 1e3;
        bump(&mut counts, &row.verdict);
        rows.push(SuiteRow {
            suite: "liquid_e8_10".into(),
            experiment: "E8".into(),
            verdict: row.verdict.clone(),
            notes: row.notes.clone(),
            elapsed_ms: elapsed,
        });
    }
    {
        let t = Instant::now();
        let (row, _) = run_experiment_9(LIQUID_SMOKE_SEED ^ 1);
        let elapsed = t.elapsed().as_secs_f64() * 1e3;
        bump(&mut counts, &row.verdict);
        rows.push(SuiteRow {
            suite: "liquid_e8_10".into(),
            experiment: "E9".into(),
            verdict: row.verdict.clone(),
            notes: row.notes.clone(),
            elapsed_ms: elapsed,
        });
    }
    {
        let t = Instant::now();
        let (row, _) = run_experiment_10(LIQUID_SMOKE_SEED ^ 2);
        let elapsed = t.elapsed().as_secs_f64() * 1e3;
        bump(&mut counts, &row.verdict);
        rows.push(SuiteRow {
            suite: "liquid_e8_10".into(),
            experiment: "E10".into(),
            verdict: row.verdict.clone(),
            notes: row.notes.clone(),
            elapsed_ms: elapsed,
        });
    }

    // ── E13 / E15 campo entrenable (smoke 1 seed) ───────────────────────────
    {
        let t = Instant::now();
        let row = run_experiment_13(FIELD_SMOKE_SEED);
        let elapsed = t.elapsed().as_secs_f64() * 1e3;
        bump(&mut counts, &row.verdict);
        rows.push(SuiteRow {
            suite: "field_e11_17".into(),
            experiment: "E13".into(),
            verdict: row.verdict.clone(),
            notes: row.notes.clone(),
            elapsed_ms: elapsed,
        });
    }
    {
        let t = Instant::now();
        let row = run_experiment_15(FIELD_SMOKE_SEED ^ 0x15);
        let elapsed = t.elapsed().as_secs_f64() * 1e3;
        bump(&mut counts, &row.verdict);
        rows.push(SuiteRow {
            suite: "field_e11_17".into(),
            experiment: "E15".into(),
            verdict: row.verdict.clone(),
            notes: row.notes.clone(),
            elapsed_ms: elapsed,
        });
    }

    // ── Clean-Room v2: un solo seed DEV (smoke hyperparams) ─────────────────
    let seed = DEV_SEEDS[0];
    notes.push(format!(
        "cleanroom_v2 smoke seed=0x{seed:X} (no confirmation 0xB300)"
    ));
    {
        let t = Instant::now();
        let suite_rows = run_smoke(seed);
        let elapsed = t.elapsed().as_secs_f64() * 1e3;
        let n = suite_rows.len().max(1) as f64;
        for r in suite_rows {
            bump(&mut counts, &r.verdict);
            rows.push(SuiteRow {
                suite: "cleanroom_v2".into(),
                experiment: r.experiment.clone(),
                verdict: r.verdict.clone(),
                notes: r.notes.clone(),
                elapsed_ms: elapsed / n,
            });
        }
    }

    ExperimentSuiteReport {
        rows,
        verdict_counts: counts,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "smoke completo E8–E15+cleanroom: decenas de segundos; correr con --ignored"]
    fn ui_suite_returns_rows() {
        let report = run_ui_experiment_suite();
        assert!(
            report.rows.len() >= 5,
            "expected ≥5 suite rows, got {}",
            report.rows.len()
        );
        assert!(!report.verdict_counts.is_empty());
        assert!(report.elapsed_ms > 0.0);
    }

    #[test]
    fn parse_suite_defaults_to_smoke() {
        assert_eq!(parse_test_suite(None), TestSuiteKind::Smoke);
        assert_eq!(parse_test_suite(Some("")), TestSuiteKind::Smoke);
        assert_eq!(parse_test_suite(Some("SMOKE")), TestSuiteKind::Smoke);
        assert_eq!(
            parse_test_suite(Some("experiments_smoke")),
            TestSuiteKind::ExperimentsSmoke
        );
        assert_eq!(
            parse_test_suite(Some("stage2_v2_dev")),
            TestSuiteKind::Stage2V2Dev
        );
        // confirmation nunca seleccionable vía API
        assert_eq!(parse_test_suite(Some("confirm")), TestSuiteKind::Smoke);
        assert_eq!(parse_test_suite(Some("0xB300")), TestSuiteKind::Smoke);
        assert_eq!(parse_test_suite(Some("bogus")), TestSuiteKind::Smoke);
        assert_eq!(TestSuiteKind::DEFAULT.as_str(), "smoke");
        assert_ne!(TestSuiteKind::DEFAULT, TestSuiteKind::Stage2V2Dev);
        assert_eq!(TestSuiteKind::Stage2V2Dev.rqm_eval(), "off");
        assert!(TestSuiteKind::Stage2V2Dev.field_only());
        assert!(!TestSuiteKind::Smoke.field_only());
    }
}
