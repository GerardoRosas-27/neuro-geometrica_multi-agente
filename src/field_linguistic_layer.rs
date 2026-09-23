//! Capa lingüística del encoder: Gemma (o una sonda con la misma forma)
//! traduce texto ↔ campo. Los tokens **no** cruzan al sustrato.
//!
//! Pila: texto → (tokenizer+capas congeladas, periférico) → features → `z`.
//! Lectura: `z` rígido → embedding de sonda → texto para el usuario.
//! `FieldState` nunca guarda `Vec<u32>`.

#![allow(clippy::needless_range_loop)]

use crate::field_encoder::{
    cluster_field_target, corpus, read_rigid_signature, write_into_field, FieldEncoder,
    LayerSkipMask, GEMMA2_LAYER_COUNT,
};
use crate::field_substrate::{
    handshake, hebb_update, ignite_casimir, ComplexT, FieldConfig, FieldState, Phasor,
};
use rand_xoshiro::rand_core::SeedableRng;

pub const HIDDEN_DIM: usize = 64;
pub const STEM_DIM: usize = 48;
const EPS: f64 = 1.0e-12;

/// Paquete de la sonda. `tokens` muere aquí.
#[derive(Clone, Debug)]
pub struct LinguisticPacket {
    tokens: Vec<u32>,
    pub hidden: Vec<f64>,
    pub layer_rms: Vec<f64>,
    pub stem_bag: Vec<f64>,
}

impl LinguisticPacket {
    pub(crate) fn from_parts(
        tokens: Vec<u32>,
        hidden: Vec<f64>,
        layer_rms: Vec<f64>,
        stem_bag: Vec<f64>,
    ) -> Self {
        Self {
            tokens,
            hidden,
            layer_rms,
            stem_bag,
        }
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Extrae features y tira los IDs. Esta es la frontera con el campo.
    pub fn into_field_features(self) -> Vec<f64> {
        let LinguisticPacket {
            tokens,
            hidden,
            layer_rms,
            stem_bag,
        } = self;
        drop(tokens);
        let mut f = layer_rms;
        f.extend_from_slice(&hidden);
        f.extend_from_slice(&stem_bag);
        let n = f.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
        for v in f.iter_mut() {
            *v /= n;
        }
        f
    }
}

pub fn linguistic_feature_dim() -> usize {
    GEMMA2_LAYER_COUNT + HIDDEN_DIM + STEM_DIM
}

/// Sonda lingüística congelada. Gemma real vive en `field_gemma_probe`.
pub trait FrozenLinguisticProbe {
    fn analyze(&mut self, text: &str) -> LinguisticPacket;
    fn name(&self) -> &'static str;
}

/// Sonda con la *forma* de Gemma 2 (26 RMS + hidden 64).
/// Usa un léxico subpalabra congelado (no n-gramas crudos) para tests sin GGUF.
pub struct GemmaShapedLexicon {
    skip: LayerSkipMask,
    seed: u64,
}

impl GemmaShapedLexicon {
    pub fn new(seed: u64) -> Self {
        Self {
            skip: LayerSkipMask::from_t21_expensive_skip(),
            seed,
        }
    }
}

fn mix(bytes: &[u8], salt: u64) -> u64 {
    let mut h = 0xC6A4_A793_5BD1_E995u64 ^ salt;
    for &b in bytes {
        h = h.wrapping_mul(0x100_0000_01B3).wrapping_add(b as u64);
        h ^= h >> 33;
    }
    h
}

/// Subpalabras toscas en español: conserva raíces mejor que bigramas.
fn subwords(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut parts = Vec::new();
    for raw in lower.split(|c: char| !c.is_alphabetic()) {
        if raw.is_empty() {
            continue;
        }
        let stem = raw
            .trim_end_matches("mente")
            .trim_end_matches("ción")
            .trim_end_matches("sión")
            .trim_end_matches("ando")
            .trim_end_matches("iendo")
            // Diminutivos ES: alinea "perrito"/"perrita" con raíz de "perro" (E8).
            .trim_end_matches("citos")
            .trim_end_matches("citas")
            .trim_end_matches("illos")
            .trim_end_matches("illas")
            .trim_end_matches("itos")
            .trim_end_matches("itas")
            .trim_end_matches("illo")
            .trim_end_matches("illa")
            .trim_end_matches("ito")
            .trim_end_matches("ita")
            .trim_end_matches("ar")
            .trim_end_matches("er")
            .trim_end_matches("ir")
            .trim_end_matches("es")
            .trim_end_matches('s')
            .trim_end_matches('o')
            .trim_end_matches('a');
        let stem = if stem.len() >= 3 { stem } else { raw };
        parts.push(stem.to_string());
        if raw.len() > stem.len() {
            parts.push(raw[stem.len()..].to_string());
        }
    }
    if parts.is_empty() {
        parts.push(lower);
    }
    parts
}

fn embed_piece(piece: &str, seed: u64) -> [f64; HIDDEN_DIM] {
    let mut v = [0.0; HIDDEN_DIM];
    for (i, slot) in v.iter_mut().enumerate() {
        let h = mix(piece.as_bytes(), seed ^ (i as u64).wrapping_mul(0x9E37));
        *slot = ((h >> 11) as f64 / (u64::MAX as f64)) * 2.0 - 1.0;
    }
    v
}

impl FrozenLinguisticProbe for GemmaShapedLexicon {
    fn analyze(&mut self, text: &str) -> LinguisticPacket {
        let pieces = subwords(text);
        // IDs sintéticos: hash de subpalabra. Nunca se copian al FieldState.
        let tokens: Vec<u32> = pieces
            .iter()
            .map(|p| (mix(p.as_bytes(), self.seed) as u32) & 0x00FF_FFFF)
            .collect();
        let mut hidden = vec![0.0; HIDDEN_DIM];
        for piece in &pieces {
            let e = embed_piece(piece, self.seed);
            for i in 0..HIDDEN_DIM {
                hidden[i] += e[i];
            }
        }
        let n = (pieces.len() as f64).max(1.0);
        for h in hidden.iter_mut() {
            *h /= n;
        }
        let hn = hidden.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
        for h in hidden.iter_mut() {
            *h /= hn;
        }
        let mut layer_rms = vec![0.0; GEMMA2_LAYER_COUNT];
        for i in 0..GEMMA2_LAYER_COUNT {
            if !self.skip.on[i] {
                continue;
            }
            let mut acc = 0.0;
            for (k, &h) in hidden.iter().enumerate() {
                let w = ((i + 1) * (k + 3)) as f64;
                acc += h * (w.sin());
            }
            layer_rms[i] = acc.abs();
        }
        let mut stem_bag = vec![0.0; STEM_DIM];
        for piece in &pieces {
            let b = (mix(piece.as_bytes(), self.seed ^ 0x57E4) as usize) % STEM_DIM;
            stem_bag[b] += 1.0;
        }
        let sn = stem_bag.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
        for v in stem_bag.iter_mut() {
            *v /= sn;
        }
        LinguisticPacket::from_parts(tokens, hidden, layer_rms, stem_bag)
    }

    fn name(&self) -> &'static str {
        "gemma-shaped-lexicon"
    }
}

pub fn skip_mask_as_enabled(skip: &LayerSkipMask) -> Vec<bool> {
    skip.on.clone()
}

/// Encoder lingüístico: sonda congelada + mapa entrenable a `z`.
pub struct LinguisticFieldCodec {
    pub probe_name: &'static str,
    pub projector: FieldEncoder,
}

impl LinguisticFieldCodec {
    pub fn new(seed: u64) -> Self {
        let n_edges = ComplexT::octahedron().n_edges();
        let feat_dim = linguistic_feature_dim();
        let mut projector = FieldEncoder::new(n_edges, seed);
        // El projector nació con feat_dim del encoder viejo; redimensionamos.
        projector.feat_dim = feat_dim;
        let scale = 0.15 / (feat_dim as f64).sqrt();
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed ^ 0x11);
        use rand::Rng;
        projector.w_re = vec![0.0; n_edges * feat_dim];
        projector.w_im = vec![0.0; n_edges * feat_dim];
        for v in projector.w_re.iter_mut().chain(projector.w_im.iter_mut()) {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * scale;
        }
        Self {
            probe_name: "unset",
            projector,
        }
    }

    pub fn write_text<P: FrozenLinguisticProbe>(
        &self,
        probe: &mut P,
        text: &str,
        psi: &mut FieldState,
    ) {
        let packet = probe.analyze(text);
        debug_assert!(!packet.tokens.is_empty() || text.trim().is_empty());
        let features = packet.into_field_features();
        let z = self.projector.encode(&features);
        write_into_field(psi, &z);
        assert!(!psi.has_token_ids(), "el sustrato no puede guardar tokens");
    }

    pub fn encode_text<P: FrozenLinguisticProbe>(&self, probe: &mut P, text: &str) -> Vec<Phasor> {
        let features = probe.analyze(text).into_field_features();
        self.projector.encode(&features)
    }
}

/// Gemma (o la sonda) traduce el campo a texto: nearest neighbor en el banco
/// de frases. Los tokens salen hacia el usuario, no hacia `Ψ`.
pub fn translate_field_to_text<P: FrozenLinguisticProbe>(
    codec: &LinguisticFieldCodec,
    probe: &mut P,
    psi: &FieldState,
    bank: &[(&str, usize)],
) -> String {
    let sig = read_rigid_signature(psi);
    let mut best = bank[0].0;
    let mut best_s = f64::NEG_INFINITY;
    for (phrase, _) in bank {
        let z = codec.encode_text(probe, phrase);
        let mut s = 0.0;
        for (e, p) in z.iter().enumerate() {
            let phase = if e < sig.len() { sig[e] } else { 0.0 };
            s += (p.arg() - phase).cos() * p.norm();
        }
        if s > best_s {
            best_s = s;
            best = phrase;
        }
    }
    best.to_string()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinguisticTrainReport {
    pub steps: usize,
    pub cluster_acc_before: f64,
    pub cluster_acc_after: f64,
    pub decode_acc_before: f64,
    pub decode_acc_after: f64,
    pub same_sim_after: f64,
    pub diff_sim_after: f64,
}

fn cluster_of_z(z: &[Phasor], n_edges: usize) -> usize {
    let mut best = 0usize;
    let mut best_s = f64::NEG_INFINITY;
    for k in 0..3 {
        let t = cluster_field_target(k, n_edges);
        let mut s = 0.0;
        for (a, b) in z.iter().zip(t.iter()) {
            s += (a.conj() * b).re;
        }
        if s > best_s {
            best_s = s;
            best = k;
        }
    }
    best
}

fn cosine_phasors(a: &[Phasor], b: &[Phasor]) -> f64 {
    let mut num = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        num += (x.conj() * y).re;
        na += x.norm_sqr();
        nb += y.norm_sqr();
    }
    num / (na.sqrt() * nb.sqrt() + EPS)
}

/// Entrena solo el projector (Gemma/sonda congelada).
pub fn train_linguistic_codec(
    steps: usize,
    seed: u64,
) -> (LinguisticFieldCodec, LinguisticTrainReport) {
    let mut probe = GemmaShapedLexicon::new(seed);
    let mut codec = LinguisticFieldCodec::new(seed);
    codec.probe_name = probe.name();
    codec.projector.eta = 0.10;
    let data = corpus();
    let n_edges = ComplexT::octahedron().n_edges();
    let cfg = FieldConfig {
        temperature: 0.0,
        ..FieldConfig::default()
    };

    let measure = |codec: &LinguisticFieldCodec, probe: &mut GemmaShapedLexicon| {
        let mut hit = 0.0;
        let mut dec = 0.0;
        let mut zs = Vec::new();
        let mut labels = Vec::new();
        for (text, k) in &data {
            let z = codec.encode_text(probe, text);
            if cluster_of_z(&z, n_edges) == *k {
                hit += 1.0;
            }
            let mut psi = FieldState::new(ComplexT::octahedron());
            write_into_field(&mut psi, &z);
            hebb_update(&mut psi, &cfg);
            let _ = handshake(&mut psi, &cfg);
            ignite_casimir(&mut psi);
            let said = translate_field_to_text(codec, probe, &psi, &data);
            let said_k = data
                .iter()
                .find(|(p, _)| *p == said)
                .map(|(_, c)| *c)
                .unwrap_or(99);
            if said_k == *k {
                dec += 1.0;
            }
            zs.push(z);
            labels.push(*k);
        }
        let n = data.len() as f64;
        let mut same = 0.0f64;
        let mut ns = 0.0f64;
        let mut diff = 0.0f64;
        let mut nd = 0.0f64;
        for i in 0..zs.len() {
            for j in (i + 1)..zs.len() {
                let s = cosine_phasors(&zs[i], &zs[j]);
                if labels[i] == labels[j] {
                    same += s;
                    ns += 1.0;
                } else {
                    diff += s;
                    nd += 1.0;
                }
            }
        }
        (hit / n, dec / n, same / ns.max(1.0), diff / nd.max(1.0))
    };

    let (acc0, dec0, _, _) = measure(&codec, &mut probe);

    use rand::Rng;
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed ^ 0x5A);
    for _ in 0..steps {
        let idx = rng.gen_range(0..data.len());
        let (text, cluster) = data[idx];
        let packet = probe.analyze(text);
        let f = packet.into_field_features();
        let z = codec.projector.encode(&f);
        let target = cluster_field_target(cluster, n_edges);
        codec.projector.delta_toward(&f, &z, &target);
    }

    let (acc1, dec1, same1, diff1) = measure(&codec, &mut probe);
    (
        codec,
        LinguisticTrainReport {
            steps,
            cluster_acc_before: acc0,
            cluster_acc_after: acc1,
            decode_acc_before: dec0,
            decode_acc_after: dec1,
            same_sim_after: same1,
            diff_sim_after: diff1,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_die_at_the_probe_boundary() {
        let mut probe = GemmaShapedLexicon::new(1);
        let codec = LinguisticFieldCodec::new(1);
        let mut psi = FieldState::new(ComplexT::octahedron());
        let packet = probe.analyze("el perro corre en el patio");
        assert!(packet.token_count() >= 2);
        codec.write_text(&mut probe, "el perro corre en el patio", &mut psi);
        assert!(!psi.has_token_ids());
        assert_eq!(psi.n(), 12);
        let dumped = format!("{psi:?}");
        assert!(!dumped.contains("token_ids"));
    }

    #[test]
    fn skip_mask_matches_gemma2_layer_count() {
        let enabled = skip_mask_as_enabled(&LayerSkipMask::from_t21_expensive_skip());
        assert_eq!(enabled.len(), GEMMA2_LAYER_COUNT);
        assert!(!enabled[2] && !enabled[17]);
        assert!(enabled[0] && enabled[25]);
    }

    #[test]
    fn linguistic_layer_improves_cluster_and_decode() {
        let (_codec, report) = train_linguistic_codec(160, 0x61A);
        println!(
            "probe=gemma-shaped steps={} cluster {:.2}->{:.2} decode {:.2}->{:.2} same {:.2} diff {:.2}",
            report.steps,
            report.cluster_acc_before,
            report.cluster_acc_after,
            report.decode_acc_before,
            report.decode_acc_after,
            report.same_sim_after,
            report.diff_sim_after
        );
        assert!(
            report.cluster_acc_after >= 0.66
                && report.cluster_acc_after + 0.05 >= report.cluster_acc_before,
            "cluster acc {report:?}"
        );
        assert!(
            report.decode_acc_after >= report.decode_acc_before || report.decode_acc_after >= 0.55,
            "decode acc {report:?}"
        );
        assert!(report.same_sim_after > report.diff_sim_after + 0.08);
    }
}
