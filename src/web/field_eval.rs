//! Evaluación completa del modelo de campo ya entrenado (post train+sueño).
//! No lanza entrenamiento infinito: solo mide el estado actual de fuse/campo/CDT.

use crate::field_substrate::{free_energy, handshake, FieldConfig, FieldState};
use crate::liquid_cdt_rqm_fuse::{FusedLiquidCdt, InferRoute};
use crate::web::experiment_suite::ExperimentSuiteReport;
use crate::web::sleep_optimize::{symmetry_score, SleepOptimizeReport};
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone, Debug, Default, Serialize)]
pub struct ConceptEvalRow {
    pub concept: usize,
    pub identity_correct: bool,
    pub shifted_correct: Option<bool>,
    pub liquid_score: f64,
    pub rqm_score: Option<f64>,
    pub route: String,
    pub latency_us: f64,
    pub engram_recall: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FieldEvalReport {
    pub engrams: usize,
    pub relational_cues: usize,
    pub had_engrams: bool,
    pub identity_accuracy: f64,
    pub shifted_accuracy: Option<f64>,
    pub identity_correct: usize,
    pub identity_total: usize,
    pub shifted_correct: usize,
    pub shifted_total: usize,
    pub liquid_latency_us_mean: f64,
    pub liquid_latency_us_p50: f64,
    pub route_histogram: HashMap<String, u64>,
    pub engram_recall_mean: Option<f64>,
    pub free_energy: Option<f64>,
    pub symmetry: Option<f64>,
    pub handshake: Option<f64>,
    pub last_sleep: Option<SleepOptimizeReport>,
    pub per_concept: Vec<ConceptEvalRow>,
    pub notes: Vec<String>,
    pub elapsed_ms: f64,
    /// Suite de experimentos UI (E8–E30 smokes); None si solo field_eval síncrono.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experiment_suite: Option<ExperimentSuiteReport>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FieldEvalStatus {
    pub running: bool,
    pub last: Option<FieldEvalReport>,
}

/// Evalúa el estado actual del fuse/campo (no entrena).
pub fn run_field_eval(
    fuse: &mut FusedLiquidCdt,
    field: Option<&FieldState>,
    cfg: &FieldConfig,
    last_sleep: Option<&SleepOptimizeReport>,
) -> FieldEvalReport {
    run_field_eval_with_progress(fuse, field, cfg, last_sleep, |_step, _total, _msg| {})
}

/// Igual que [`run_field_eval`] con progreso por concepto (consola en vivo).
pub fn run_field_eval_with_progress(
    fuse: &mut FusedLiquidCdt,
    field: Option<&FieldState>,
    cfg: &FieldConfig,
    last_sleep: Option<&SleepOptimizeReport>,
    mut on_progress: impl FnMut(usize, usize, &str),
) -> FieldEvalReport {
    let t0 = Instant::now();
    let mut notes = Vec::new();
    let n = fuse.num_labels.max(1);
    let cands: Vec<usize> = (0..n).collect();
    let engrams = fuse.engram_count();
    let had = engrams > 0;
    if !had {
        notes.push("sin engramas: se evalúa igualmente el estado actual".into());
    }
    on_progress(
        0,
        n,
        &format!("iniciando batería ({n} conceptos, engramas={engrams})"),
    );

    let mut rows = Vec::new();
    let mut id_ok = 0usize;
    let mut id_tot = 0usize;
    let mut sh_ok = 0usize;
    let mut sh_tot = 0usize;
    let mut lats = Vec::new();
    let mut routes: HashMap<String, u64> = HashMap::new();
    let mut recall_scores = Vec::new();

    for concept in 0..n {
        on_progress(concept + 1, n, &format!("evaluando concepto {concept}/{n}"));
        let t1 = Instant::now();
        let rep = fuse.infer(concept, &cands);
        let lat = t1.elapsed().as_secs_f64() * 1e6;
        lats.push(lat);
        let route = match rep.route {
            InferRoute::Liquid => "Liquid",
            InferRoute::RqmFallback => "RqmFallback",
        };
        *routes.entry(route.to_string()).or_insert(0) += 1;
        let id_correct = rep.predicted == concept;
        id_tot += 1;
        if id_correct {
            id_ok += 1;
        }

        let mut shifted_correct = None;
        if !fuse.relational_cues.is_empty() {
            let r2 = fuse.infer_with_force(concept, &cands, true);
            let ok = if fuse.relational_cues.contains(&concept) {
                matches!(r2.route, InferRoute::RqmFallback) || r2.rqm_score.is_some()
            } else {
                true
            };
            shifted_correct = Some(ok);
            sh_tot += 1;
            if ok {
                sh_ok += 1;
            }
        }

        let recall = fuse.memory.engrams.get(&concept).map(|e| {
            let s: f64 = e.template.iter().map(|x| x * x).sum::<f64>().sqrt();
            (s / (1.0 + s)).clamp(0.0, 1.0)
        });
        if let Some(r) = recall {
            recall_scores.push(r);
        }

        rows.push(ConceptEvalRow {
            concept,
            identity_correct: id_correct,
            shifted_correct,
            liquid_score: rep.liquid_score,
            rqm_score: rep.rqm_score,
            route: route.into(),
            latency_us: lat,
            engram_recall: recall,
        });
    }

    lats.sort_by(|a, b| a.total_cmp(b));
    let mean_lat = if lats.is_empty() {
        0.0
    } else {
        lats.iter().sum::<f64>() / lats.len() as f64
    };
    let p50 = if lats.is_empty() {
        0.0
    } else {
        lats[lats.len() / 2]
    };

    let (fe, sym, hs) = if let Some(psi) = field {
        let f = free_energy(psi, cfg).total;
        let s = symmetry_score(psi);
        let mut psi2 = psi.clone();
        let h = handshake(&mut psi2, cfg);
        (Some(f), Some(s), Some(h))
    } else {
        notes.push("sin FieldState en AppState: métricas de campo omitidas".into());
        (None, None, None)
    };

    if let Some(sleep) = last_sleep {
        notes.push(format!(
            "último sueño: ΔF={:.4} Δsym={:.4} pruned={}",
            sleep.free_energy_after - sleep.free_energy_before,
            sleep.symmetry_after - sleep.symmetry_before,
            sleep.routes_pruned
        ));
    }

    on_progress(n, n, "batería completada");

    FieldEvalReport {
        engrams,
        relational_cues: fuse.relational_cues.len(),
        had_engrams: had,
        identity_accuracy: if id_tot == 0 {
            0.0
        } else {
            id_ok as f64 / id_tot as f64
        },
        shifted_accuracy: if sh_tot == 0 {
            None
        } else {
            Some(sh_ok as f64 / sh_tot as f64)
        },
        identity_correct: id_ok,
        identity_total: id_tot,
        shifted_correct: sh_ok,
        shifted_total: sh_tot,
        liquid_latency_us_mean: mean_lat,
        liquid_latency_us_p50: p50,
        route_histogram: routes,
        engram_recall_mean: if recall_scores.is_empty() {
            None
        } else {
            Some(recall_scores.iter().sum::<f64>() / recall_scores.len() as f64)
        },
        free_energy: fe,
        symmetry: sym,
        handshake: hs,
        last_sleep: last_sleep.cloned(),
        per_concept: rows,
        notes,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        experiment_suite: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_substrate::ComplexT;

    #[test]
    fn eval_runs_without_engrams() {
        let mut fuse = FusedLiquidCdt::new(8);
        let cfg = FieldConfig::default();
        let field = FieldState::new(ComplexT::octahedron());
        let report = run_field_eval(&mut fuse, Some(&field), &cfg, None);
        assert_eq!(report.identity_total, 8);
        assert!(!report.had_engrams);
        let v = serde_json::to_value(&report).unwrap();
        assert!(v.get("identity_accuracy").is_some());
        assert!(v.get("per_concept").is_some());
        assert!(v.get("route_histogram").is_some());
    }

    #[test]
    fn eval_after_teach_and_sleep() {
        let mut fuse = FusedLiquidCdt::new(8);
        let cands: Vec<usize> = (0..8).collect();
        for i in 0..8 {
            fuse.observe(i, &cands);
            fuse.teach_relation(i, (i + 3) % 8);
        }
        let _ = fuse.sleep_consolidate();
        let cfg = FieldConfig::default();
        let report = run_field_eval(&mut fuse, None, &cfg, None);
        assert!(report.engrams > 0);
        assert!(report.identity_total > 0);
        assert!(report.relational_cues > 0);
    }
}
