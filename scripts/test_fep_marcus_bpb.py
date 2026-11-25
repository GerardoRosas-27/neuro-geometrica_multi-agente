#!/usr/bin/env python3
"""
Suite de Pruebas Unitarias e Invariantes Científicos para la Medición Real de
BPB, BPC, Cross-Entropy Loss (nats) y Perplejidad (PPL) en AURA v3 Genesis + FEP-MARCUS v3.

Verifica 8 invariantes estrictos:
  1. Identidades exactas de teoría de la información:
     BPB = CE / ln(2), BPC = BPB * (N_bytes / N_chars), PPL_byte = exp(CE) = 2^BPB, PPL_char = 2^BPC.
  2. Normalización estricta del simplex Delta^255: max_t |sum_b P(b | b_{<t}) - 1| < 1e-5, 0 NaN/Inf.
  3. Causalidad estricta (sin lookahead): mutar bytes futuros b_{>=t} produce 0.0 cambio en P(. | b_{<t}).
  4. Cero fuga Train / Held-Out: 0 archivos compartidos, 0 SHA-256 compartidos, 0 bloques 128B compartidos,
     0 solapamiento 64B con `canonical_eval_md` / `canonical_shift_rust`, 0 bytes del repo en prenatal.
  5. Evaluación estrictamente congelada: `evaluate_sequence_frozen` y `evaluate_blind_bayesian_stream`
     y `evaluate_sequence_online_causal` NO mutan el estado del modelo congelado (huella idéntica).
  6. Ley de Dale estricta (W^E, W^I >= 0, 0 violaciones) antes y después de entrenar y podar por microglía.
  7. Auditoría FepEnergyTester 100% PASS en los 4 invariantes LM + convergencia monótona FEP.
  8. Integridad criptográfica `AUDIT-v3 PASS` (SHA-256 de aura-engine/crates/*.rs 100% intacto).
"""

import math
import os
import subprocess
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fep_marcus_core import (  # noqa: E402
    AuraFepCorticalLanguageModel,
    BigramByteBaseline,
    DaleMountcastleNeuralByteColumn,
    FepEnergyTester,
    RepoTextCorpusSplitter,
    compute_exact_byte_lm_metrics,
    generate_synthetic_prenatal_byte_waves,
)


def test_exact_information_theoretic_identities() -> None:
    # Caso 1: distribución uniforme exacta sobre UTF-8 con caracteres multibyte en español
    sample = "Energía Libre Variacional: ΔF ≤ 0, precisión π y causalidad do(X)!\n".encode("utf-8")
    n = len(sample)
    n_chars = len(sample.decode("utf-8"))
    assert n > n_chars, "Se esperaban caracteres UTF-8 multibyte para probar BPC > BPB"

    unif = np.full((n, 256), 1.0 / 256.0, dtype=np.float64)
    m = compute_exact_byte_lm_metrics(unif, sample, dale_violations=0, frozen_verified=True)
    assert abs(m["ce_nats"] - math.log(256.0)) < 1e-5
    assert abs(m["bpb"] - 8.0) < 1e-5
    assert abs(m["bpc"] - 8.0 * (n / n_chars)) < 1e-5
    assert abs(m["ppl_byte"] - 256.0) < 1e-4
    assert abs(m["ppl_char"] - (2.0 ** m["bpc"])) < 1e-3
    assert m["max_prob_norm_error"] < 1e-12
    assert m["nan_inf_count"] == 0

    # Caso 2: distribución no uniforme arbitraria normalizada
    rng = np.random.default_rng(123)
    logits = rng.normal(0.0, 2.0, size=(n, 256))
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    m2 = compute_exact_byte_lm_metrics(p, sample, dale_violations=0, frozen_verified=True)
    assert abs(m2["bpb"] - m2["ce_nats"] / math.log(2.0)) < 1e-5
    assert abs(m2["bpc"] - m2["bpb"] * (n / n_chars)) < 1e-5
    assert abs(m2["ppl_byte"] - math.exp(m2["ce_nats"])) / max(1.0, m2["ppl_byte"]) < 1e-5
    assert abs(m2["ppl_char"] - (2.0 ** m2["bpc"])) / max(1.0, m2["ppl_char"]) < 1e-5
    assert m2["max_prob_norm_error"] < 1e-12
    print("[PASS] test_exact_information_theoretic_identities")


def test_strict_causality_no_lookahead() -> None:
    train_toy = (
        b"## Arquitectura AURA v3 Genesis + FEP-MARCUS v3 (1.7021 BPB)\n"
        b"La energia libre variacional F(q, o) acota la sorpresa -ln p(o).\n"
    ) * 20
    seq_a = b"## Arquitectura AURA v3 Genesis mide BPB y PPL en texto real 1.70."
    cut = 30
    seq_b = seq_a[:cut] + (b"Z" * (len(seq_a) - cut))
    assert len(seq_a) == len(seq_b)

    for genesis_flag in (True, False):
        lm = AuraFepCorticalLanguageModel(seed=42, enable_aura_v3_genesis=genesis_flag)
        lm.run_prenatal_in_utero_byte_waves(n_bytes=2048)
        lm.specialists["markdown"].fit_bytes(train_toy)
        lm.shared_cortex.fit_bytes(train_toy)
        if genesis_flag:
            assert lm.shared_digit_norm is not None
            assert lm.shared_casedigit_norm is not None
            assert lm.shared_lex_skip is not None
            assert lm.ca3_long_match is not None
            lm.shared_digit_norm.fit_bytes(train_toy)
            lm.shared_casedigit_norm.fit_bytes(train_toy)
            lm.shared_lex_skip.fit_bytes(train_toy)
            lm.ca3_long_match.fit_bytes(train_toy)
            lm.domain_neurals["markdown"].train_cyclic_epoch(train_toy, n_samples=2048, batch_size=256, lr=5e-3)
        else:
            lm.neural_column.train_cyclic_epoch(train_toy, n_samples=2048, batch_size=256, lr=5e-3)

        pa, _, _ = lm.predict_domain_probs_256(seq_a, domain="markdown", enable_myelination=True)
        pb, _, _ = lm.predict_domain_probs_256(seq_b, domain="markdown", enable_myelination=True)

        # En todas las posiciones t <= cut, b_{<t} es idéntico -> P(. | b_{<t}) debe ser bit-idéntico!
        max_diff_prefix = float(np.max(np.abs(pa[:cut + 1] - pb[:cut + 1])))
        assert max_diff_prefix == 0.0, f"Fuga causal detectada (genesis={genesis_flag}): max_diff={max_diff_prefix}"

        max_norm = float(np.max(np.abs(pa.sum(axis=1) - 1.0)))
        assert max_norm < 1e-5, f"Error de normalización excesivo (genesis={genesis_flag}): {max_norm}"
        assert lm._total_dale_violations() == 0
    print("[PASS] test_strict_causality_no_lookahead (Arm 1 & Arm 2 8-expert)")


def test_anti_leakage_splitter_and_frozen_eval() -> None:
    splitter = RepoTextCorpusSplitter(
        ROOT_DIR, n_cycles=8, seed=42, eval_bytes_per_domain=4096, cal_bytes_per_domain=2048
    )
    al = splitter.audit_report
    assert al["shared_files"] == 0, f"Archivos compartidos: {al['shared_files']}"
    assert al["shared_hashes"] == 0, f"Hashes compartidos: {al['shared_hashes']}"
    assert al["shared_128b_blocks"] == 0, f"Bloques 128B compartidos: {al['shared_128b_blocks']}"
    assert al["canonical_64b_overlap"] == 0, f"Solapamiento canónico 64B: {al['canonical_64b_overlap']}"
    assert al["prenatal_repo_bytes"] == 0
    assert al["total_train_files"] > 100
    assert al["total_held_out_files"] > 20
    assert al["canonical_eval_md_bytes"] == 6144
    assert al["canonical_shift_rust_bytes"] == 6144

    # Verificar que eval congelado, blind routing y online causal NO mutan estado del modelo
    lm = AuraFepCorticalLanguageModel(seed=42, enable_aura_v3_genesis=True)
    lm.run_prenatal_in_utero_byte_waves(n_bytes=2048)
    lm.train_cycle_shard(0, splitter, neural_samples=4096, calibrate_on_train_cal=True)

    fp_before = lm.state_fingerprint()
    m_fr = lm.evaluate_sequence_frozen(splitter.held_out_buffers["markdown"][:2048], domain="markdown")
    m_bl = lm.evaluate_blind_bayesian_stream(
        {d: splitter.held_out_buffers[d][:1024] for d in lm.DOMAINS}, max_bytes_per_doc=1024
    )
    m_on = lm.evaluate_sequence_online_causal(splitter.canonical_shift_rust[:1024], domain="rust")
    fp_after = lm.state_fingerprint()

    assert fp_before == fp_after, "La evaluación mutó el estado congelado del modelo!"
    assert m_fr["frozen_verified"] is True
    assert m_bl["frozen_verified"] is True
    assert m_on["frozen_verified"] is True
    assert m_fr["dale_violations"] == 0
    assert m_fr["nan_inf_count"] == 0

    tester = FepEnergyTester()
    aud = tester.audit_language_eval(m_fr, anti_leakage=al)
    assert all(aud.values()), f"Auditoría FepEnergyTester falló: {aud}"
    print("[PASS] test_anti_leakage_splitter_and_frozen_eval")


def test_dale_law_and_microglia_pruning() -> None:
    for ms_flag in (False, True):
        col = DaleMountcastleNeuralByteColumn(seed=42, d_hid=256 if ms_flag else 192, enable_multiscale_l4=ms_flag)
        assert col.dale_violations() == 0
        waves = generate_synthetic_prenatal_byte_waves(seed=42, n_bytes=4096)
        col.train_cyclic_epoch(waves, n_samples=4096, batch_size=512, lr=5e-3)
        assert col.dale_violations() == 0
        st = col.microglia_prune(threshold=0.02)
        assert st["dale_violations"] == 0
        assert st["pruned_pct"] > 50.0, f"Se esperaba poda física > 50%, obtenido {st['pruned_pct']}%"
    print(f"[PASS] test_dale_law_and_microglia_pruning (multiscale_l4=False/True, pruned={st['pruned_pct']}%, dale_v=0)")


def test_bigram_baseline_proper() -> None:
    bg = BigramByteBaseline(alpha=0.25)
    data = b"pub fn step(&mut self, x: f32) -> f32 { x * 0.5 }\n" * 100
    bg.fit_bytes(data)
    m = bg.evaluate(data)
    assert m["max_prob_norm_error"] < 1e-12
    assert 0.1 < m["bpb"] < 3.0
    print(f"[PASS] test_bigram_baseline_proper (bpb={m['bpb']:.4f})")


def test_audit_v3_pass_preserved() -> None:
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT_DIR, "aura-engine", "freeze_and_verify_v3.py"), "audit"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout.strip()
    assert "[AUDIT-v3 PASS]" in out, f"AUDIT-v3 no pasó: {out}"
    assert "cc90b95faa8a1885e308b81270a94babcb57b624a7323599edb8b38f0f366171" in out
    print(f"[PASS] test_audit_v3_pass_preserved ({out})")


if __name__ == "__main__":
    test_exact_information_theoretic_identities()
    test_strict_causality_no_lookahead()
    test_dale_law_and_microglia_pruning()
    test_bigram_baseline_proper()
    test_anti_leakage_splitter_and_frozen_eval()
    test_audit_v3_pass_preserved()
    print("\nALL 6/6 SCIENTIFIC INVARIANT TESTS PASSED!")
