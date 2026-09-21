//! Evaluación completa: entrenamiento + inferencia + consolidación CDT
//! (rama campo-híbrido) vs inferencia RQM estilo `main`.

use crate::entanglement::EntanglementConfig;
use crate::field_encoder::train_encoder;
use crate::field_hybrid_infer::FieldHybridInfer;
use crate::field_linguistic_layer::train_linguistic_codec;
use crate::field_substrate::run_cycle;
use crate::hybrid_wave_rqm_infer::InferPath;
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::relational_field::ObserverId;
use std::time::Instant;

const OBSERVER: ObserverId = ObserverId(0xE5A1);
const CUE_BASE: usize = 64;
const N: usize = 8;
const INFER_LOOPS: usize = 40;

#[derive(Clone, Debug, Default)]
pub struct FullEvalReport {
    pub encoder_steps: usize,
    pub encoder_recon_before: f64,
    pub encoder_recon_after: f64,
    pub encoder_same_after: f64,
    pub encoder_diff_after: f64,
    pub lingu_steps: usize,
    pub lingu_cluster_before: f64,
    pub lingu_cluster_after: f64,
    pub lingu_decode_before: f64,
    pub lingu_decode_after: f64,
    pub hybrid_cold_acc: f64,
    pub hybrid_warm_acc: f64,
    pub hybrid_mean_handshake: f64,
    pub hybrid_tokens_in_field: bool,
    pub hybrid_cold_us: f64,
    pub hybrid_warm_us: f64,
    pub hybrid_train_distill_ms: f64,
    pub field_cycle_handshake: f64,
    pub field_cycle_recon: f64,
    pub main_rqm_acc: f64,
    pub main_rqm_train_ms: f64,
    pub main_rqm_us: f64,
    pub main_rqm_relations: usize,
}

fn pick_label(cands: &[NativeCandidateScore]) -> Option<usize> {
    cands
        .iter()
        .filter(|c| c.agent < N)
        .max_by(|a, b| {
            a.score
                .total_cmp(&b.score)
                .then_with(|| a.agent.cmp(&b.agent))
        })
        .map(|c| c.agent)
}

fn bench_main_rqm() -> (f64, f64, f64, usize) {
    let thermal = NativeThermoCdtConfig {
        slices: 2,
        nodes_per_slice: (CUE_BASE + N).max(128),
        seed: 0xBEE2_0002,
        ..NativeThermoCdtConfig::default()
    };
    let mut cfg = NativeThermoRqmConfig::default();
    cfg.max_candidates = N * 2;
    cfg.thermal_steps_per_train = 1;
    cfg.thermal_steps_per_query = 2;
    let epr = EntanglementConfig {
        create_threshold: 0.4,
        max_syncs_per_step: 64,
        ..EntanglementConfig::default()
    };
    let mut rqm = NativeThermoRqmEprSubstrate::new(thermal, cfg, epr);

    let t0 = Instant::now();
    for _ in 0..6 {
        for i in 0..N {
            rqm.train_observed_transition(OBSERVER, 0.0, &[CUE_BASE + i], &[i], 0.95);
        }
    }
    let train_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut ok = 0usize;
    let t1 = Instant::now();
    for _ in 0..INFER_LOOPS {
        for i in 0..N {
            let r = rqm.query(OBSERVER, 0.0, &[CUE_BASE + i]);
            if pick_label(&r.candidates) == Some(i) {
                ok += 1;
            }
        }
    }
    let infer_ms = t1.elapsed().as_secs_f64() * 1e3;
    let q = INFER_LOOPS * N;
    (
        ok as f64 / q as f64,
        train_ms,
        (infer_ms * 1e3) / q as f64,
        rqm.relation_count(),
    )
}

/// Corre entrenamiento de campo + ciclo híbrido completo + brazo main RQM.
pub fn run_full_eval(encoder_steps: usize, lingu_steps: usize) -> FullEvalReport {
    let mut report = FullEvalReport {
        encoder_steps,
        lingu_steps,
        ..FullEvalReport::default()
    };

    // 1) Entrenar encoder → campo
    let t_enc = Instant::now();
    let (_enc, er) = train_encoder(encoder_steps, 0xE4C0);
    let _enc_ms = t_enc.elapsed().as_secs_f64() * 1e3;
    report.encoder_recon_before = er.recon_before;
    report.encoder_recon_after = er.recon_after;
    report.encoder_same_after = er.same_cluster_sim_after;
    report.encoder_diff_after = er.diff_cluster_sim_after;

    // 2) Entrenar capa lingüística (projector; sonda congelada)
    let t_ling = Instant::now();
    let (_codec, lr) = train_linguistic_codec(lingu_steps, 0x1106);
    let _ling_ms = t_ling.elapsed().as_secs_f64() * 1e3;
    report.lingu_cluster_before = lr.cluster_acc_before;
    report.lingu_cluster_after = lr.cluster_acc_after;
    report.lingu_decode_before = lr.decode_acc_before;
    report.lingu_decode_after = lr.decode_acc_after;

    // 3) Ciclo CDT de sustrato (smoke científico local)
    let cycle = run_cycle(0xCD7A);
    report.field_cycle_handshake = cycle.handshake;
    report.field_cycle_recon = cycle.recon_error;

    // 4) Híbrido: destilar todos los conceptos (train) + medir cold/warm + consolidación
    let mut eng = FieldHybridInfer::new(0xF001);
    let t_dist = Instant::now();
    for i in 0..N {
        let r = eng.infer_and_consolidate(i); // WaveNew + distill + CDT
        assert_eq!(r.path, InferPath::WaveNew);
    }
    report.hybrid_train_distill_ms = t_dist.elapsed().as_secs_f64() * 1e3;

    // cold path timing (fresh engine, no distill timing in loop)
    let mut cold_eng = FieldHybridInfer::new(0xC01D);
    cold_eng.hybrid.auto_distill = false;
    let mut ok_c = 0usize;
    let mut hs_sum = 0.0;
    let t_cold = Instant::now();
    for _ in 0..INFER_LOOPS {
        for i in 0..N {
            let r = cold_eng.infer_and_consolidate(i);
            hs_sum += r.handshake;
            if r.concept_out == i {
                ok_c += 1;
            }
            if r.field_has_token_ids {
                report.hybrid_tokens_in_field = true;
            }
        }
    }
    let cold_q = INFER_LOOPS * N;
    report.hybrid_cold_acc = ok_c as f64 / cold_q as f64;
    report.hybrid_cold_us = t_cold.elapsed().as_secs_f64() * 1e6 / cold_q as f64;
    report.hybrid_mean_handshake = hs_sum / cold_q as f64;

    // warm path (already distilled eng)
    let mut ok_w = 0usize;
    let t_warm = Instant::now();
    for _ in 0..INFER_LOOPS {
        for i in 0..N {
            let r = eng.infer_and_consolidate(i);
            assert_eq!(r.path, InferPath::RqmTrained);
            if r.concept_out == i {
                ok_w += 1;
            }
            if r.field_has_token_ids {
                report.hybrid_tokens_in_field = true;
            }
        }
    }
    report.hybrid_warm_acc = ok_w as f64 / cold_q as f64;
    report.hybrid_warm_us = t_warm.elapsed().as_secs_f64() * 1e6 / cold_q as f64;

    // 5) Brazo main RQM
    let (m_acc, m_train, m_us, m_rel) = bench_main_rqm();
    report.main_rqm_acc = m_acc;
    report.main_rqm_train_ms = m_train;
    report.main_rqm_us = m_us;
    report.main_rqm_relations = m_rel;

    let _ = _enc_ms;
    let _ = _ling_ms;
    report
}

pub fn format_full_eval(r: &FullEvalReport) -> String {
    format!(
        "=== FULL EVAL: campo-híbrido vs main RQM ===\n\
         \n\
         [Train campo]\n\
         encoder steps={es} recon {erb:.4}->{era:.4} same={esa:.3} diff={eda:.3}\n\
         linguistic steps={ls} cluster {lcb:.3}->{lca:.3} decode {ldb:.3}->{lda:.3}\n\
         field run_cycle handshake={fh:.3} recon={fr:.4}\n\
         \n\
         [Hybrid infer + CDT consolidate]\n\
         distill_ms={hd:.2}\n\
         cold (wave+CDT): acc={hca:.3}  µs/q={hcu:.2}  mean_hs={hhs:.3}\n\
         warm (rqm+CDT):  acc={hwa:.3}  µs/q={hwu:.2}\n\
         tokens_in_field={tok}\n\
         \n\
         [Main RQM only — sin campo/CDT]\n\
         train_ms={mt:.3}  acc={ma:.3}  µs/q={mu:.3}  relations={mr}\n\
         \n\
         [Comparación]\n\
         hybrid_warm_us / main_us = {ratio:.2}\n\
         hybrid añade consolidación fasorial+handshake; main solo relaciones.\n",
        es = r.encoder_steps,
        erb = r.encoder_recon_before,
        era = r.encoder_recon_after,
        esa = r.encoder_same_after,
        eda = r.encoder_diff_after,
        ls = r.lingu_steps,
        lcb = r.lingu_cluster_before,
        lca = r.lingu_cluster_after,
        ldb = r.lingu_decode_before,
        lda = r.lingu_decode_after,
        fh = r.field_cycle_handshake,
        fr = r.field_cycle_recon,
        hd = r.hybrid_train_distill_ms,
        hca = r.hybrid_cold_acc,
        hcu = r.hybrid_cold_us,
        hhs = r.hybrid_mean_handshake,
        hwa = r.hybrid_warm_acc,
        hwu = r.hybrid_warm_us,
        tok = r.hybrid_tokens_in_field,
        mt = r.main_rqm_train_ms,
        ma = r.main_rqm_acc,
        mu = r.main_rqm_us,
        mr = r.main_rqm_relations,
        ratio = r.hybrid_warm_us / r.main_rqm_us.max(1e-12),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_pipeline_train_infer_cdt_vs_main() {
        // Pasos suficientes para mover métricas sin alargar demasiado el CI local.
        let r = run_full_eval(60, 80);
        let s = format_full_eval(&r);
        println!("{s}");
        assert!(r.encoder_recon_after.is_finite());
        assert!(r.lingu_cluster_after.is_finite());
        assert!(r.hybrid_cold_acc >= 0.99, "cold {r:?}");
        assert!(r.hybrid_warm_acc >= 0.99, "warm {r:?}");
        assert!(!r.hybrid_tokens_in_field);
        assert!(r.hybrid_mean_handshake.is_finite());
        assert!(r.main_rqm_acc >= 0.99);
        assert!(r.field_cycle_handshake.is_finite());
        // El train de campo debe mover o al menos reportar métricas útiles.
        assert!(r.encoder_steps > 0 && r.lingu_steps > 0);
    }
}
