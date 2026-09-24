# Confirmación — Etapa 2 Clean-Room v3 (0xB300–0xB30F)

Rama: `exp/field-autonomy-next`. Commit tip: `ce282a20b47292aeefc9e2316a9604af0c3af8a5`.
Hyperparams: same `HyperparamLock::long()` locked on DEV; no retune.
Wall time: **1145.4s**. Filas: 240.

## Histogramas

| Experimento | verdict hist |
|---|---|
| `E18C_cdt_conditions_C0_C6` | NULL:2, PARTIAL:14 |
| `E18_reinforced_rule_learning` | POSITIVE:7, STRONG_POSITIVE:9 |
| `E19_provenance_audit` | PARTIAL:1, POSITIVE:1, STRONG_POSITIVE:14 |
| `E20_rule_vs_trajectory_antimem` | PARTIAL:16 |
| `E21_interpolation_vs_extrapolation` | PARTIAL:4, POSITIVE:6, STRONG_POSITIVE:6 |
| `E22_composition_rqm_off` | NULL:1, PARTIAL:3, POSITIVE:9, STRONG_POSITIVE:3 |
| `E23_rollout_no_teacher_forcing` | NULL:1, PARTIAL:12, POSITIVE:2, STRONG_POSITIVE:1 |
| `E24_static_vs_dynamic_paired` | POSITIVE:16 |
| `E25_central_experience_cdt_cycle` | NULL:2, PARTIAL:14 |
| `E26_experience_ablation_A_G` | NULL:2, PARTIAL:14 |
| `E27_rule_transfer` | NEGATIVE:1, NULL:2, PARTIAL:8, POSITIVE:3, STRONG_POSITIVE:2 |
| `E28_continual_forgetting` | PARTIAL:16 |
| `E29_causal_intervention` | PARTIAL:16 |
| `E30_serialize_reload_persistence` | PARTIAL:16 |
| `FASE_A_cleanroom_field_only` | POSITIVE:16 |

CSV: [`resultados_etapa_2_v3_confirm.csv`](resultados_etapa_2_v3_confirm.csv)

## Lectura honesta (confirmación)

Hyperparams frozen from DEV lock (`HyperparamLock::long()`). No retune after seeing 0xB*.

| Exp | Confirm hist | POS+STRONG |
|---|---|---|
| E18 | POSITIVE:7, STRONG_POSITIVE:9 | 16/16 |
| E21 | PARTIAL:4, POSITIVE:6, STRONG_POSITIVE:6 | 12/16 |
| E22 | NULL:1, PARTIAL:3, POSITIVE:9, STRONG_POSITIVE:3 | 12/16 |
| E24 | POSITIVE:16 | 16/16 |
| E23 | NULL:1, PARTIAL:12, POSITIVE:2, STRONG_POSITIVE:1 | 3/16 |

**Confirmation: YES** — all four gates majority POS+STRONG on held-out 0xB300–0xB30F.
