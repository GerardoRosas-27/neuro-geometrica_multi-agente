//! Puente: campo sin tokens + híbrido onda/RQM + LLM solo periferia de decode.
//!
//! Flujo:
//! 1. (Opcional) texto → `LinguisticPacket.into_field_features()` — tira token IDs.
//! 2. Features → `content_id` discreto (sin vocabulario en el sustrato).
//! 3. Cue **nuevo** → `WavePredictCore`; cue **entrenado** → RQM nativo.
//! 4. Predicción → fasores en `FieldState` → handshake + Hebb (CDT geométrico).
//! 5. Decode periférico: concepto → texto (sonda); **nunca** escribe tokens en Ψ.

use crate::field_encoder::{write_into_field, FieldEncoder};
use crate::field_linguistic_layer::{FrozenLinguisticProbe, LinguisticPacket};
use crate::field_substrate::{handshake, hebb_update, ComplexT, FieldConfig, FieldState, Phasor};
use crate::hybrid_wave_rqm_infer::{HybridReport, HybridWaveRqm, InferPath};
use std::f64::consts::PI;

const NUM_CONCEPTS: usize = 8;
/// Decodificador periférico: concepto → texto. No toca `FieldState`.
pub trait PeripheralDecoder {
    fn decode_concept(&self, concept: usize) -> String;
}

/// Léxico mínimo de etiquetas (periferia). Sustituible por Gemma decode real.
#[derive(Clone, Debug, Default)]
pub struct LabelDecoder {
    pub labels: Vec<&'static str>,
}

impl LabelDecoder {
    pub fn concepts8() -> Self {
        Self {
            labels: vec![
                "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
            ],
        }
    }
}

impl PeripheralDecoder for LabelDecoder {
    fn decode_concept(&self, concept: usize) -> String {
        let i = concept % self.labels.len().max(1);
        self.labels[i].to_string()
    }
}

/// Cuantiza features de campo a un id de concepto (0..NUM_CONCEPTS).
pub fn features_to_concept_id(features: &[f64], num_concepts: usize) -> usize {
    if features.is_empty() || num_concepts == 0 {
        return 0;
    }
    // Proyección estable: ángulo del primer plano (f0,f1) + energía.
    let a = features.first().copied().unwrap_or(0.0);
    let b = features.get(1).copied().unwrap_or(0.0);
    let ang = b.atan2(a); // (-π, π]
    let u = (ang + PI) / (2.0 * PI); // [0,1]
    let e: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mix = (0.7 * u + 0.3 * (e / (1.0 + e))).clamp(0.0, 0.999999);
    (mix * num_concepts as f64).floor() as usize % num_concepts
}

/// Concepto → patrón de fasores en aristas (escritura al campo, sin tokens).
pub fn concept_to_phasors(concept: usize, n_edges: usize) -> Vec<Phasor> {
    let c = concept % NUM_CONCEPTS;
    let base = c as f64 * (2.0 * PI / NUM_CONCEPTS as f64);
    (0..n_edges)
        .map(|e| {
            let phase = base + 0.35 * (e as f64);
            Phasor::from_polar(1.0, phase)
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct FieldHybridInfer {
    pub hybrid: HybridWaveRqm,
    pub field: FieldState,
    pub cfg: FieldConfig,
    pub encoder: FieldEncoder,
    pub num_concepts: usize,
    pub decoder: LabelDecoder,
}

#[derive(Clone, Debug)]
pub struct FieldHybridReport {
    pub path: InferPath,
    pub concept_in: usize,
    pub concept_out: usize,
    pub hybrid: HybridReport,
    pub handshake: f64,
    pub decoded: String,
    pub field_has_token_ids: bool,
}

impl FieldHybridInfer {
    pub fn new(seed: u64) -> Self {
        let t = ComplexT::octahedron();
        let n = t.n_edges();
        Self {
            hybrid: HybridWaveRqm::new(NUM_CONCEPTS),
            field: FieldState::new(t),
            cfg: FieldConfig::default(),
            encoder: FieldEncoder::new(n, seed),
            num_concepts: NUM_CONCEPTS,
            decoder: LabelDecoder::concepts8(),
        }
    }

    /// Texto → features (firewall) → concept id. Tokens no llegan al campo.
    pub fn observe_text<P: FrozenLinguisticProbe>(&mut self, probe: &mut P, text: &str) -> usize {
        let packet: LinguisticPacket = probe.analyze(text);
        assert!(
            packet.token_count() > 0 || text.is_empty(),
            "probe may tokenize internally"
        );
        let features = packet.into_field_features();
        features_to_concept_id(&features, self.num_concepts)
    }

    /// Observación ya como concepto (pista tokenless / entrenamiento de campo).
    pub fn observe_concept(&self, concept: usize) -> usize {
        concept % self.num_concepts
    }

    /// Consolida la predicción en el campo: escribe fasores + handshake + Hebb.
    pub fn consolidate_prediction(&mut self, concept: usize) -> f64 {
        let z = concept_to_phasors(concept, self.field.n());
        write_into_field(&mut self.field, &z);
        // Marca rígido tras lock.
        for m in self.field.m.iter_mut() {
            *m = 1.0;
        }
        let hs = handshake(&mut self.field, &self.cfg);
        hebb_update(&mut self.field, &self.cfg);
        hs
    }

    /// Inferencia híbrida + consolidación fasorial + decode periférico.
    pub fn infer_and_consolidate(&mut self, concept_in: usize) -> FieldHybridReport {
        let concept_in = self.observe_concept(concept_in);
        let hy = self.hybrid.infer(concept_in);
        let hs = self.consolidate_prediction(hy.predicted);
        let decoded = self.decoder.decode_concept(hy.predicted);
        FieldHybridReport {
            path: hy.path,
            concept_in,
            concept_out: hy.predicted,
            hybrid: hy,
            handshake: hs,
            decoded,
            field_has_token_ids: self.field.has_token_ids(),
        }
    }

    /// Atajo: texto → (firewall) → híbrido → campo → texto periférico.
    pub fn infer_from_text<P: FrozenLinguisticProbe>(
        &mut self,
        probe: &mut P,
        text: &str,
    ) -> FieldHybridReport {
        let c = self.observe_text(probe, text);
        self.infer_and_consolidate(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_linguistic_layer::GemmaShapedLexicon;
    use crate::hybrid_wave_rqm_infer::InferPath;

    #[test]
    fn features_never_put_tokens_in_field() {
        let mut eng = FieldHybridInfer::new(0xF1E1D);
        let mut probe = GemmaShapedLexicon::new(7);
        let r = eng.infer_from_text(&mut probe, "hola campo sin tokens");
        assert!(!r.field_has_token_ids);
        assert!(!eng.field.has_token_ids());
        assert!(!r.decoded.is_empty());
        println!(
            "text→hybrid path={:?} in={} out={} hs={:.3} decode={}",
            r.path, r.concept_in, r.concept_out, r.handshake, r.decoded
        );
    }

    #[test]
    fn cold_wave_then_warm_rqm_on_field_stack() {
        let mut eng = FieldHybridInfer::new(0xC01D);
        let r1 = eng.infer_and_consolidate(2);
        assert_eq!(r1.path, InferPath::WaveNew);
        assert_eq!(r1.concept_out, 2);
        let r2 = eng.infer_and_consolidate(2);
        assert_eq!(r2.path, InferPath::RqmTrained);
        assert_eq!(r2.concept_out, 2);
        assert!(!eng.field.has_token_ids());
        println!(
            "field cold={:?} warm={:?} decode={}",
            r1.path, r2.path, r2.decoded
        );
    }

    #[test]
    fn consolidate_writes_phasors_and_handshake() {
        let mut eng = FieldHybridInfer::new(0xCD7);
        let hs = eng.consolidate_prediction(4);
        assert!(hs.is_finite());
        assert_eq!(eng.field.z.len(), eng.field.n());
        let amp: f64 = eng.field.z.iter().map(|z| z.norm()).sum::<f64>() / eng.field.n() as f64;
        assert!(amp > 0.5, "phasors should be written: amp={amp}");
        assert!(!eng.field.has_token_ids());
    }

    #[test]
    fn peripheral_decode_does_not_touch_field_tokens() {
        let eng = FieldHybridInfer::new(1);
        let before = eng.field.has_token_ids();
        let s = eng.decoder.decode_concept(3);
        assert_eq!(s, "delta");
        assert_eq!(eng.field.has_token_ids(), before);
    }

    #[test]
    fn concept_id_stable_for_same_features() {
        let f = vec![0.2, 0.5, 0.1, -0.3];
        let a = features_to_concept_id(&f, 8);
        let b = features_to_concept_id(&f, 8);
        assert_eq!(a, b);
    }
}
