//! Inferencia experimental: entrada → colapso de ondas de spin → RQM nativo.
//!
//! Solo en la rama `exp/fluido-3d-astra`. No modifica el RQM de producción en
//! `main`; reutiliza `NativeThermoRqmEprSubstrate` como motor relacional.
//!
//! Flujo:
//!   1. Codificar la entrada como paquete ψ en `SpinFluid3D`.
//!   2. Evolucionar NLS enfocante (colapso / interferencia local).
//!   3. Discretizar el colapso a nodos-feature.
//!   4. Entrenar / consultar RQM nativo (causa=features → efecto=etiqueta).

use crate::entanglement::EntanglementConfig;
use crate::fluid3d_astra::N;
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeRqmQueryReport, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::relational_field::ObserverId;
use crate::spin_fluid3d::SpinFluid3D;

/// Etiquetas / conceptos objetivo (nodos RQM 0..NUM_LABELS).
pub const NUM_LABELS: usize = 4;
/// Base de nodos que describen el colapso de ψ.
pub const FEATURE_BASE: usize = 16;
/// Rejilla gruesa 4³ → 64 celdas de pico.
pub const COARSE: usize = 4;
pub const SPATIAL_FEATURES: usize = COARSE * COARSE * COARSE; // 64
pub const AMP_BINS: usize = 8;
pub const PHASE_BINS: usize = 8;
pub const FEATURE_END: usize = FEATURE_BASE + SPATIAL_FEATURES + AMP_BINS + PHASE_BINS; // 96

const OBSERVER: ObserverId = ObserverId(0x5F10);

#[derive(Clone, Debug)]
pub struct SpinFluidRqmInfer {
    pub rqm: NativeThermoRqmEprSubstrate,
    pub collapse_steps: usize,
    pub alpha: f64,
    pub beta: f64,
    pub dt: f64,
    pub packet_amplitude: f64,
    pub packet_sigma: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CollapseFeatures {
    pub peak: [f64; 3],
    pub max_amp: f64,
    pub phase: f64,
    pub norm: f64,
    pub spatial_node: usize,
    pub amp_node: usize,
    pub phase_node: usize,
}

#[derive(Clone, Debug)]
pub struct InferReport {
    pub input: usize,
    pub features: CollapseFeatures,
    pub feature_nodes: Vec<usize>,
    pub query: NativeRqmQueryReport,
    pub predicted: Option<usize>,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainEpochReport {
    pub pairs: usize,
    pub relations: usize,
}

impl SpinFluidRqmInfer {
    pub fn new() -> Self {
        let thermal = NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice: FEATURE_END.max(128),
            seed: 0x5F10_F100,
            ..NativeThermoCdtConfig::default()
        };
        let mut rqm_cfg = NativeThermoRqmConfig::default();
        rqm_cfg.max_candidates = NUM_LABELS * 4;
        rqm_cfg.thermal_steps_per_train = 1;
        rqm_cfg.thermal_steps_per_query = 2;
        rqm_cfg.thermal_activation_margin = 0.05;
        let epr = EntanglementConfig {
            create_threshold: 0.4,
            max_syncs_per_step: 64,
            ..EntanglementConfig::default()
        };
        Self {
            rqm: NativeThermoRqmEprSubstrate::new(thermal, rqm_cfg, epr),
            collapse_steps: 28,
            alpha: 0.35,
            beta: 2.5,
            dt: 0.02,
            packet_amplitude: 0.82,
            packet_sigma: 1.55,
        }
    }

    /// Codifica un id de entrada (0..NUM_LABELS) como paquete de spin distinto.
    pub fn encode_input(&self, input: usize) -> SpinFluid3D {
        let input = input % NUM_LABELS;
        let mut spin = SpinFluid3D::new(self.alpha, self.beta, self.dt);
        let mid = (N / 2) as f64;
        // Centros y momentos distintos por clase → colapsos en regiones distintas.
        let offsets = [
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let momenta = [
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.0, 0.15, 0.0],
            [0.0, 0.0, 0.12],
        ];
        let o = offsets[input];
        let k = momenta[input];
        let amp = self.packet_amplitude * (1.0 + 0.04 * input as f64);
        spin.add_gaussian_packet(
            [mid + o[0], mid + o[1], mid + o[2]],
            self.packet_sigma,
            amp,
            k,
        );
        spin
    }

    pub fn collapse(&self, spin: &mut SpinFluid3D) {
        spin.step_n(self.collapse_steps, None);
    }

    pub fn extract_features(&self, spin: &SpinFluid3D) -> CollapseFeatures {
        let peak = spin.peak_coords();
        let max_amp = spin.max_amplitude();
        let idx = spin.peak_index();
        let phase = spin.psi[idx].arg();
        let cx = ((peak[0] / N as f64) * COARSE as f64).floor() as usize;
        let cy = ((peak[1] / N as f64) * COARSE as f64).floor() as usize;
        let cz = ((peak[2] / N as f64) * COARSE as f64).floor() as usize;
        let cx = cx.min(COARSE - 1);
        let cy = cy.min(COARSE - 1);
        let cz = cz.min(COARSE - 1);
        let spatial = FEATURE_BASE + cx + COARSE * (cy + COARSE * cz);
        let amp_bin = ((max_amp / 2.0) * AMP_BINS as f64).floor() as usize;
        let amp_bin = amp_bin.min(AMP_BINS - 1);
        let amp_node = FEATURE_BASE + SPATIAL_FEATURES + amp_bin;
        let phase01 = (phase + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
        let phase_bin = (phase01.clamp(0.0, 0.999) * PHASE_BINS as f64).floor() as usize;
        let phase_node = FEATURE_BASE + SPATIAL_FEATURES + AMP_BINS + phase_bin;
        CollapseFeatures {
            peak,
            max_amp,
            phase,
            norm: spin.l2_norm_sq(),
            spatial_node: spatial,
            amp_node,
            phase_node,
        }
    }

    pub fn feature_nodes(feat: &CollapseFeatures) -> Vec<usize> {
        let mut v = vec![feat.spatial_node, feat.amp_node, feat.phase_node];
        v.sort_unstable();
        v.dedup();
        v
    }

    fn observer_phase(feat: &CollapseFeatures) -> f32 {
        feat.phase as f32
    }

    /// Simula colapso de la entrada y refuerza features → etiqueta con RQM nativo.
    pub fn train_pair(&mut self, input: usize, label: usize, success: f32) -> CollapseFeatures {
        let label = label % NUM_LABELS;
        let mut spin = self.encode_input(input);
        self.collapse(&mut spin);
        let feat = self.extract_features(&spin);
        let cause = Self::feature_nodes(&feat);
        let effect = [label];
        self.rqm.train_observed_transition(
            OBSERVER,
            Self::observer_phase(&feat),
            &cause,
            &effect,
            success.clamp(0.0, 1.0),
        );
        feat
    }

    pub fn train_identity_epoch(&mut self, repeats: usize) -> TrainEpochReport {
        let mut pairs = 0usize;
        for _ in 0..repeats {
            for i in 0..NUM_LABELS {
                let _ = self.train_pair(i, i, 0.95);
                pairs += 1;
            }
        }
        TrainEpochReport {
            pairs,
            relations: self.rqm.relation_count(),
        }
    }

    /// Dada una entrada: colapsa ψ y consulta RQM nativo sobre los features.
    pub fn infer(&mut self, input: usize) -> InferReport {
        let mut spin = self.encode_input(input);
        self.collapse(&mut spin);
        let feat = self.extract_features(&spin);
        let feature_nodes = Self::feature_nodes(&feat);
        let query = self
            .rqm
            .query(OBSERVER, Self::observer_phase(&feat), &feature_nodes);
        let (predicted, confidence) = pick_label(&query.candidates);
        InferReport {
            input: input % NUM_LABELS,
            features: feat,
            feature_nodes,
            query,
            predicted,
            confidence,
        }
    }

    /// Exactitud sobre las NUM_LABELS entradas identidad.
    pub fn identity_accuracy(&mut self) -> f64 {
        let mut ok = 0usize;
        for i in 0..NUM_LABELS {
            let r = self.infer(i);
            if r.predicted == Some(i) {
                ok += 1;
            }
        }
        ok as f64 / NUM_LABELS as f64
    }
}

impl Default for SpinFluidRqmInfer {
    fn default() -> Self {
        Self::new()
    }
}

fn pick_label(candidates: &[NativeCandidateScore]) -> (Option<usize>, f32) {
    let mut best: Option<&NativeCandidateScore> = None;
    for c in candidates {
        if c.agent >= NUM_LABELS {
            continue;
        }
        match best {
            None => best = Some(c),
            Some(b) if c.score > b.score => best = Some(c),
            _ => {}
        }
    }
    match best {
        Some(c) => (Some(c.agent), c.score),
        None => (None, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_inputs_collapse_to_distinct_spatial_features() {
        let eng = SpinFluidRqmInfer::new();
        let mut nodes = Vec::new();
        for i in 0..NUM_LABELS {
            let mut spin = eng.encode_input(i);
            eng.collapse(&mut spin);
            let f = eng.extract_features(&spin);
            println!(
                "in={i} peak={:?} amp={:.3} spatial={}",
                f.peak, f.max_amp, f.spatial_node
            );
            nodes.push(f.spatial_node);
            assert!(f.max_amp > 0.5, "collapse should keep amplitude");
        }
        let mut uniq = nodes.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert!(
            uniq.len() >= 3,
            "inputs should map to mostly distinct spatial bins: {nodes:?}"
        );
    }

    #[test]
    fn rqm_learns_collapse_features_to_labels() {
        let mut eng = SpinFluidRqmInfer::new();
        let before = eng.identity_accuracy();
        let train = eng.train_identity_epoch(6);
        let after = eng.identity_accuracy();
        println!(
            "rqm-infer before={before:.2} after={after:.2} pairs={} rel={}",
            train.pairs, train.relations
        );
        for i in 0..NUM_LABELS {
            let r = eng.infer(i);
            println!(
                "  infer in={i} pred={:?} conf={:.3} feats={:?}",
                r.predicted, r.confidence, r.feature_nodes
            );
        }
        assert!(
            train.relations > 0,
            "training should create RQM relations"
        );
        assert!(
            after >= 0.75,
            "after training, collapse→RQM should recover labels: before={before} after={after}"
        );
        assert!(
            after > before + 0.24,
            "training should improve accuracy: before={before} after={after}"
        );
    }

    #[test]
    fn rqm_can_learn_non_identity_mapping() {
        let mut eng = SpinFluidRqmInfer::new();
        // input i → label (i+1) % 4
        for _ in 0..8 {
            for i in 0..NUM_LABELS {
                let _ = eng.train_pair(i, (i + 1) % NUM_LABELS, 0.95);
            }
        }
        let mut ok = 0usize;
        for i in 0..NUM_LABELS {
            let r = eng.infer(i);
            let want = (i + 1) % NUM_LABELS;
            println!("shift in={i} pred={:?} want={want}", r.predicted);
            if r.predicted == Some(want) {
                ok += 1;
            }
        }
        let acc = ok as f64 / NUM_LABELS as f64;
        assert!(
            acc >= 0.75,
            "shifted mapping should be learnable via collapse features: acc={acc}"
        );
    }
}
