//! Instantáneas de telemetría para paneles líquido / CDT / RQM / entrenamiento.

use crate::liquid_cdt_memory::SleepReport;
use crate::liquid_cdt_rqm_fuse::{FuseReport, InferRoute};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Clone, Debug, Default, Serialize)]
pub struct TelemetrySnapshot {
    pub liquid_queries: u64,
    pub liquid_score_sum: f64,
    pub liquid_score_last: f64,
    pub liquid_latency_us_last: f64,
    pub liquid_latency_us_sum: f64,
    pub engrams: usize,
    pub wake_buffer: usize,
    pub rqm_infer_calls: u64,
    pub rqm_train_calls: u64,
    pub route_liquid: u64,
    pub route_rqm: u64,
    pub sleeps: u64,
    pub train_epochs_done: u64,
    pub train_correct: u64,
    pub train_total: u64,
    /// Histograma de rutas: "Liquid" | "RqmFallback" → count
    pub route_histogram: HashMap<String, u64>,
}

impl TelemetrySnapshot {
    pub fn record_fuse(&mut self, report: &FuseReport, latency_us: f64) {
        self.liquid_queries = self.liquid_queries.wrapping_add(1);
        self.liquid_score_sum += report.liquid_score;
        self.liquid_score_last = report.liquid_score;
        self.liquid_latency_us_last = latency_us;
        self.liquid_latency_us_sum += latency_us;
        match report.route {
            InferRoute::Liquid => {
                self.route_liquid = self.route_liquid.wrapping_add(1);
                *self.route_histogram.entry("Liquid".into()).or_insert(0) += 1;
            }
            InferRoute::RqmFallback => {
                self.route_rqm = self.route_rqm.wrapping_add(1);
                *self
                    .route_histogram
                    .entry("RqmFallback".into())
                    .or_insert(0) += 1;
            }
        }
    }

    pub fn record_sleep(&mut self, _sleep: &SleepReport, engrams: usize, wake: usize) {
        self.sleeps = self.sleeps.wrapping_add(1);
        self.engrams = engrams;
        self.wake_buffer = wake;
    }

    pub fn sync_fuse_counters(
        &mut self,
        rqm_infer: u64,
        rqm_train: u64,
        engrams: usize,
        wake: usize,
    ) {
        self.rqm_infer_calls = rqm_infer;
        self.rqm_train_calls = rqm_train;
        self.engrams = engrams;
        self.wake_buffer = wake;
    }

    pub fn liquid_score_avg(&self) -> f64 {
        if self.liquid_queries == 0 {
            0.0
        } else {
            self.liquid_score_sum / self.liquid_queries as f64
        }
    }

    pub fn liquid_route_pct(&self) -> f64 {
        let t = self.route_liquid + self.route_rqm;
        if t == 0 {
            0.0
        } else {
            100.0 * self.route_liquid as f64 / t as f64
        }
    }

    pub fn rqm_route_pct(&self) -> f64 {
        let t = self.route_liquid + self.route_rqm;
        if t == 0 {
            0.0
        } else {
            100.0 * self.route_rqm as f64 / t as f64
        }
    }

    pub fn train_accuracy(&self) -> Option<f64> {
        if self.train_total == 0 {
            None
        } else {
            Some(self.train_correct as f64 / self.train_total as f64)
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FuseReportDto {
    pub observation: usize,
    pub predicted: usize,
    pub liquid_score: f64,
    pub route: String,
    pub rqm_score: Option<f64>,
    pub top1_score: f64,
    pub top2_score: f64,
    pub margin: f64,
    pub energy: f64,
    pub abstained: bool,
    pub hops: usize,
}

impl From<&FuseReport> for FuseReportDto {
    fn from(r: &FuseReport) -> Self {
        Self {
            observation: r.observation,
            predicted: r.predicted,
            liquid_score: r.liquid_score,
            route: match r.route {
                InferRoute::Liquid => "Liquid".into(),
                InferRoute::RqmFallback => "RqmFallback".into(),
            },
            rqm_score: r.rqm_score,
            top1_score: r.top1_score,
            top2_score: r.top2_score,
            margin: r.margin,
            energy: r.energy,
            abstained: r.abstained,
            hops: r.hops,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SleepReportDto {
    pub episodes_consolidated: usize,
    pub engrams_before: usize,
    pub engrams_after: usize,
    pub used_rqm: bool,
    pub rqm_relations_trained: usize,
    pub sleep_ms: f64,
}

impl From<&SleepReport> for SleepReportDto {
    fn from(s: &SleepReport) -> Self {
        Self {
            episodes_consolidated: s.episodes_consolidated,
            engrams_before: s.engrams_before,
            engrams_after: s.engrams_after,
            used_rqm: s.used_rqm,
            rqm_relations_trained: s.rqm_relations_trained,
            sleep_ms: s.sleep_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liquid_cdt_rqm_fuse::InferRoute;

    #[test]
    fn histogram_and_pct() {
        let mut m = TelemetrySnapshot::default();
        let r = FuseReport {
            observation: 1,
            predicted: 1,
            liquid_score: 0.9,
            route: InferRoute::Liquid,
            rqm_score: None,
            top1_score: 0.9,
            top2_score: 0.4,
            margin: 0.5,
            energy: 0.10536051565782635,
            abstained: false,
            hops: 1,
        };
        m.record_fuse(&r, 1.5);
        assert_eq!(m.route_liquid, 1);
        assert!((m.liquid_route_pct() - 100.0).abs() < 1e-9);
    }
}
