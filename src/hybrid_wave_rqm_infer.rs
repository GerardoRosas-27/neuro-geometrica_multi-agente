//! Híbrido de inferencia (rama aislada desde `main`):
//!
//! - **Nuevo** (cue aún no consolidado): líquido de ondas
//!   (`WavePredictCore` — interferencia pasado×futuro).
//! - **Ya entrenado**: RQM nativo (`NativeThermoRqmEprSubstrate`).
//!
//! Tras una predicción por ondas con score suficiente, se **destila** la
//! transición cue→label en RQM para que las siguientes consultas usen RQM.

use crate::entanglement::EntanglementConfig;
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::relational_field::ObserverId;
use crate::wave_predict_core::{PredictionReport, WavePredictCore};
use std::collections::HashMap;
use std::time::Instant;

const OBSERVER: ObserverId = ObserverId(0x41B1);
const CUE_BASE: usize = 64;
const DEFAULT_LABELS: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferPath {
    /// Primera vez / cue frío → líquido de ondas.
    WaveNew,
    /// Cue ya destilado → RQM.
    RqmTrained,
}

#[derive(Clone, Debug)]
pub struct HybridReport {
    pub path: InferPath,
    pub observation: usize,
    pub predicted: usize,
    pub score: f64,
    pub distilled: bool,
    pub trained_cues: usize,
}

#[derive(Clone, Debug)]
pub struct HybridWaveRqm {
    pub wave: WavePredictCore,
    pub rqm: NativeThermoRqmEprSubstrate,
    /// observation → label consolidado en RQM.
    pub trained: HashMap<usize, usize>,
    pub distill_min_score: f64,
    pub auto_distill: bool,
    pub num_labels: usize,
}

impl Default for HybridWaveRqm {
    fn default() -> Self {
        Self::new(DEFAULT_LABELS)
    }
}

impl HybridWaveRqm {
    pub fn new(num_labels: usize) -> Self {
        let thermal = NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice: (CUE_BASE + num_labels).max(128),
            seed: 0x41B1_F100,
            ..NativeThermoCdtConfig::default()
        };
        let mut rqm_cfg = NativeThermoRqmConfig::default();
        rqm_cfg.max_candidates = num_labels * 2;
        rqm_cfg.thermal_steps_per_train = 1;
        rqm_cfg.thermal_steps_per_query = 2;
        rqm_cfg.thermal_activation_margin = 0.05;
        let epr = EntanglementConfig {
            create_threshold: 0.4,
            max_syncs_per_step: 64,
            ..EntanglementConfig::default()
        };
        Self {
            wave: WavePredictCore::new(),
            rqm: NativeThermoRqmEprSubstrate::new(thermal, rqm_cfg, epr),
            trained: HashMap::new(),
            distill_min_score: 0.2,
            auto_distill: true,
            num_labels,
        }
    }

    pub fn is_trained(&self, observation: usize) -> bool {
        self.trained.contains_key(&(observation % self.num_labels))
    }

    fn cue(observation: usize) -> usize {
        CUE_BASE + observation
    }

    fn candidates(&self) -> Vec<usize> {
        (0..self.num_labels).collect()
    }

    fn pick_rqm_label(cands: &[NativeCandidateScore], n: usize) -> Option<usize> {
        cands
            .iter()
            .filter(|c| c.agent < n)
            .max_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.agent.cmp(&b.agent)))
            .map(|c| c.agent)
    }

    /// Destila una predicción de ondas en RQM (marca el cue como entrenado).
    pub fn distill(&mut self, observation: usize, label: usize, success: f32) {
        let obs = observation % self.num_labels;
        let lab = label % self.num_labels;
        self.rqm.train_observed_transition(
            OBSERVER,
            0.0,
            &[Self::cue(obs)],
            &[lab],
            success.clamp(0.0, 1.0),
        );
        self.trained.insert(obs, lab);
    }

    /// Inferencia **nueva**: siempre líquido de ondas (no consulta RQM).
    pub fn infer_new(&mut self, observation: usize) -> HybridReport {
        let obs = observation % self.num_labels;
        let cands = self.candidates();
        let pred: PredictionReport = self.wave.predict_from_observation(obs, &cands);
        let mut distilled = false;
        if self.auto_distill && pred.best_score >= self.distill_min_score {
            self.distill(obs, pred.best_content, 0.95);
            distilled = true;
        }
        HybridReport {
            path: InferPath::WaveNew,
            observation: obs,
            predicted: pred.best_content,
            score: pred.best_score,
            distilled,
            trained_cues: self.trained.len(),
        }
    }

    /// Inferencia **ya entrenada**: solo RQM (falla limpio si el cue es frío).
    pub fn infer_trained(&mut self, observation: usize) -> Option<HybridReport> {
        let obs = observation % self.num_labels;
        if !self.is_trained(obs) {
            return None;
        }
        let report = self.rqm.query(OBSERVER, 0.0, &[Self::cue(obs)]);
        let predicted = Self::pick_rqm_label(&report.candidates, self.num_labels)?;
        let score = report
            .candidates
            .iter()
            .find(|c| c.agent == predicted)
            .map(|c| c.score as f64)
            .unwrap_or(0.0);
        Some(HybridReport {
            path: InferPath::RqmTrained,
            observation: obs,
            predicted,
            score,
            distilled: false,
            trained_cues: self.trained.len(),
        })
    }

    /// Router: si el cue está entrenado → RQM; si no → ondas (+ destilado opcional).
    pub fn infer(&mut self, observation: usize) -> HybridReport {
        let obs = observation % self.num_labels;
        if self.is_trained(obs) {
            self.infer_trained(obs).unwrap_or_else(|| self.infer_new(obs))
        } else {
            self.infer_new(obs)
        }
    }

    /// Precalienta RQM con un mapa identidad (simula “ya entrenado”).
    pub fn pretrain_identity(&mut self, repeats: usize) {
        for _ in 0..repeats {
            for i in 0..self.num_labels {
                self.distill(i, i, 0.95);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HybridBench {
    pub cold_wave_us: f64,
    pub warm_rqm_us: f64,
    pub cold_acc: f64,
    pub warm_acc: f64,
    pub speedup_warm_vs_cold: f64,
}

pub fn bench_hybrid(loops: usize) -> HybridBench {
    let n = DEFAULT_LABELS;
    // Cold: todo por ondas (sin destilar para medir solo wave)
    let mut cold = HybridWaveRqm::new(n);
    cold.auto_distill = false;
    let mut ok_c = 0usize;
    let t0 = Instant::now();
    for _ in 0..loops {
        for i in 0..n {
            let r = cold.infer_new(i);
            if r.predicted == i {
                ok_c += 1;
            }
        }
    }
    let cold_ms = t0.elapsed().as_secs_f64() * 1e3;
    let cold_q = loops * n;
    let cold_wave_us = (cold_ms * 1e3) / cold_q as f64;

    // Warm: pretrain + solo RQM
    let mut warm = HybridWaveRqm::new(n);
    warm.pretrain_identity(4);
    let mut ok_w = 0usize;
    let t1 = Instant::now();
    for _ in 0..loops {
        for i in 0..n {
            let r = warm.infer(i);
            assert_eq!(r.path, InferPath::RqmTrained);
            if r.predicted == i {
                ok_w += 1;
            }
        }
    }
    let warm_ms = t1.elapsed().as_secs_f64() * 1e3;
    let warm_rqm_us = (warm_ms * 1e3) / cold_q as f64;

    HybridBench {
        cold_wave_us,
        warm_rqm_us,
        cold_acc: ok_c as f64 / cold_q as f64,
        warm_acc: ok_w as f64 / cold_q as f64,
        speedup_warm_vs_cold: cold_wave_us / warm_rqm_us.max(1e-12),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_uses_wave_then_distills_to_rqm() {
        let mut h = HybridWaveRqm::new(8);
        assert!(!h.is_trained(3));
        let r1 = h.infer(3);
        assert_eq!(r1.path, InferPath::WaveNew);
        assert_eq!(r1.predicted, 3);
        assert!(r1.distilled);
        assert!(h.is_trained(3));

        let r2 = h.infer(3);
        assert_eq!(r2.path, InferPath::RqmTrained);
        assert_eq!(r2.predicted, 3);
        println!(
            "hybrid cold→wave pred={} then warm→rqm pred={} trained={}",
            r1.predicted, r2.predicted, r2.trained_cues
        );
    }

    #[test]
    fn infer_trained_none_when_cold() {
        let mut h = HybridWaveRqm::new(8);
        assert!(h.infer_trained(1).is_none());
    }

    #[test]
    fn pretrain_routes_all_to_rqm() {
        let mut h = HybridWaveRqm::new(8);
        h.pretrain_identity(3);
        for i in 0..8 {
            let r = h.infer(i);
            assert_eq!(r.path, InferPath::RqmTrained);
            assert_eq!(r.predicted, i);
        }
    }

    #[test]
    fn hybrid_cold_wave_and_warm_rqm_accuracies() {
        let b = bench_hybrid(40);
        println!(
            "hybrid cold_wave={:.3}µs acc={:.3} | warm_rqm={:.3}µs acc={:.3} | warm/cold path note: warm_us/cold_us={:.3}",
            b.cold_wave_us, b.cold_acc, b.warm_rqm_us, b.warm_acc, b.warm_rqm_us / b.cold_wave_us.max(1e-12)
        );
        assert!(b.cold_acc >= 0.99);
        assert!(b.warm_acc >= 0.99);
    }

    #[test]
    fn distill_then_infer_trained_works() {
        let mut h = HybridWaveRqm::new(8);
        h.auto_distill = false;
        let wave = h.infer_new(5);
        assert_eq!(wave.path, InferPath::WaveNew);
        h.distill(5, wave.predicted, 0.95);
        let rqm = h.infer_trained(5).expect("should be trained");
        assert_eq!(rqm.path, InferPath::RqmTrained);
        assert_eq!(rqm.predicted, 5);
    }
}
