//! Telemetría del campo v4 (chat, lotes, sueños).

use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct TelemetrySnapshot {
    /// Pasos del campo servidos por el chat ON.
    pub field_queries: u64,
    pub field_correct: u64,
    pub field_rel_err_last: f64,
    pub field_rel_err_sum: f64,
    pub field_latency_us_last: f64,
    pub train_batches: u64,
    pub sleeps: u64,
}

impl TelemetrySnapshot {
    pub fn record_field(&mut self, rel_err: f64, correct: bool, latency_us: f64) {
        self.field_queries += 1;
        self.field_correct += u64::from(correct);
        self.field_rel_err_last = rel_err;
        self.field_rel_err_sum += rel_err.min(10.0);
        self.field_latency_us_last = latency_us;
    }

    pub fn field_rel_err_avg(&self) -> f64 {
        if self.field_queries == 0 {
            0.0
        } else {
            self.field_rel_err_sum / self.field_queries as f64
        }
    }
}
