# Resultados — Etapa 2 autonomía de campo

Protocolo: `docs/etapa_2_autonomia_campo_experimentos.md`
Módulo: `src/field_autonomy_stage2.rs`
Rama: `exp/field-autonomy-stage-2`
Seeds desarrollo: 8 × `0xE1800..0xE1807`
Periferia: **numeric field-only** (GGUF presente en disco pero **no** usado en el núcleo de reglas).

## Implementado vs diferido

| Estado | Experimentos |
|--------|--------------|
| Implementado (Fase A+B + parcial C/D) | FASE_A_field_only_scaffold, E18A_direct_no_cdt, E18B_episodic_cdt_consolidation, E18C_cdt_vs_no_cdt, E19_no_lookup_audit, E20_trajectory_vs_rule, E21_abstract_rule_param, E22_unobserved_composition, E23_long_horizon, E24_static_vs_dynamic, E25_cdt_experience_not_answer, E26_experience_ablation |
| Diferido (scaffold INVALID) | E27_rule_transfer, E28_continual_forgetting, E29_causal_intervention, E30_persistence_restart |

## Resumen de veredictos (8 seeds)

| Experimento | mean dyn | mean static | mean leak | verdict histogram |
|-------------|---------:|------------:|----------:|-------------------|
| FASE_A_field_only_scaffold | 0.000 | 0.000 | 0.0 | POSITIVE:8 |
| E18A_direct_no_cdt | 0.546 | 0.265 | 0.0 | NEGATIVE:2, NULL:2, PARTIAL:1, POSITIVE:2, STRONG_POSITIVE:1 |
| E18B_episodic_cdt_consolidation | 0.563 | 0.428 | 0.0 | NEGATIVE:4, NULL:1, POSITIVE:2, STRONG_POSITIVE:1 |
| E18C_cdt_vs_no_cdt | 0.626 | 0.478 | 0.0 | NEGATIVE:1, NULL:1, PARTIAL:3, POSITIVE:3 |
| E19_no_lookup_audit | 0.546 | 0.265 | 0.0 | POSITIVE:8 |
| E20_trajectory_vs_rule | 0.294 | 0.829 | 0.0 | NEGATIVE:6, PARTIAL:2 |
| E21_abstract_rule_param | 0.393 | 0.195 | 0.0 | NULL:5, PARTIAL:3 |
| E22_unobserved_composition | 0.533 | 0.346 | 0.0 | NEGATIVE:1, NULL:4, PARTIAL:3 |
| E23_long_horizon | 0.829 | 0.323 | 0.0 | NULL:1, PARTIAL:6, POSITIVE:1 |
| E24_static_vs_dynamic | 0.133 | 0.733 | 0.0 | NEGATIVE:7, POSITIVE:1 |
| E25_cdt_experience_not_answer | 0.235 | 0.389 | 0.0 | NEGATIVE:5, NULL:1, PARTIAL:1, STRONG_POSITIVE:1 |
| E26_experience_ablation | 0.536 | 0.565 | 0.0 | NEGATIVE:4, PARTIAL:2, POSITIVE:2 |
| E27_rule_transfer | 0.000 | 0.000 | 0.0 | INVALID:8 |
| E28_continual_forgetting | 0.000 | 0.000 | 0.0 | INVALID:8 |
| E29_causal_intervention | 0.000 | 0.000 | 0.0 | INVALID:8 |
| E30_persistence_restart | 0.000 | 0.000 | 0.0 | INVALID:8 |

## Lectura honesta

- Fase A: FIELD_ONLY + auditoría de provenance cableada; RQM/CDT OFF en path de eval; leakage=0 en suite.
- E18A: aprendizaje directo de traslación; señal positiva en varias semillas, alta varianza con 4 pares (+aug).
- E18B: consolidación episódica; target de test no se recupera de CDT (queries eval=0).
- E18C: C0/C1/C2/C3; experience_gain reportado; no siempre C2>C1 (resultado mixto, no forzado).
- E19: auditoría no-lookup sobre path E18A; contadores CDT/RQM/table/NN/attractor = 0.
- E20–E24: implementados; veredictos mixtos (PARTIAL/NULL/NEGATIVE frecuentes) — no se afirma regla universal.
- E25–E26: scaffolds reutilizando E18B/E18C.
- E27–E30: diferidos explícitamente (`INVALID`).

## Controles

Cada E18* reporta static / linear / NN / table donde aplica. Table en estados novel ≈0. NN es memoria, no regla.

## Tabla cruda

| Exp | seed | mode | rule_gen | dyn | static | leak | verdict |
|-----|-----:|------|---------:|----:|-------:|-----:|---------|
| FASE_A_field_only_scaffold | 923648 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923648 | MODE_2_DYNAMIC_FIELD | 0.947 | 0.947 | -0.448 | 0 | STRONG_POSITIVE |
| E18B_episodic_cdt_consolidation | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.876 | 0.876 | 0.375 | 0 | POSITIVE |
| E18C_cdt_vs_no_cdt | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.463 | 0.463 | 0.193 | 0 | POSITIVE |
| E19_no_lookup_audit | 923648 | MODE_2_DYNAMIC_FIELD | 0.947 | 0.947 | -0.448 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923648 | MODE_2_DYNAMIC_FIELD | 0.747 | 0.747 | 0.635 | 0 | PARTIAL |
| E21_abstract_rule_param | 923648 | MODE_2_DYNAMIC_FIELD | 0.189 | 0.189 | -0.023 | 0 | NULL |
| E22_unobserved_composition | 923648 | MODE_2_DYNAMIC_FIELD | 0.504 | 0.504 | 0.556 | 0 | NULL |
| E23_long_horizon | 923648 | MODE_2_DYNAMIC_FIELD | 0.243 | 0.899 | 0.243 | 0 | PARTIAL |
| E24_static_vs_dynamic | 923648 | MODE_2_DYNAMIC_FIELD | 0.531 | 0.531 | 0.569 | 0 | NEGATIVE |
| E25_cdt_experience_not_answer | 923173 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.016 | 0.016 | 0.977 | 0 | NEGATIVE |
| E26_experience_ablation | 923174 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.929 | 0.929 | 0.756 | 0 | POSITIVE |
| E27_rule_transfer | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923649 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923649 | MODE_2_DYNAMIC_FIELD | -0.340 | -0.340 | 0.577 | 0 | NEGATIVE |
| E18B_episodic_cdt_consolidation | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.900 | 0.900 | -0.512 | 0 | POSITIVE |
| E18C_cdt_vs_no_cdt | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.351 | 0.351 | 0.097 | 0 | POSITIVE |
| E19_no_lookup_audit | 923649 | MODE_2_DYNAMIC_FIELD | -0.340 | -0.340 | 0.577 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923649 | MODE_2_DYNAMIC_FIELD | 0.559 | 0.559 | 0.845 | 0 | NEGATIVE |
| E21_abstract_rule_param | 923649 | MODE_2_DYNAMIC_FIELD | 0.772 | 0.772 | 0.385 | 0 | PARTIAL |
| E22_unobserved_composition | 923649 | MODE_2_DYNAMIC_FIELD | 0.417 | 0.417 | 0.194 | 0 | NULL |
| E23_long_horizon | 923649 | MODE_2_DYNAMIC_FIELD | 0.396 | 0.587 | 0.396 | 0 | NULL |
| E24_static_vs_dynamic | 923649 | MODE_2_DYNAMIC_FIELD | -0.110 | -0.110 | 0.740 | 0 | NEGATIVE |
| E25_cdt_experience_not_answer | 923172 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.649 | 0.649 | -0.262 | 0 | NULL |
| E26_experience_ablation | 923175 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.920 | 0.920 | 0.198 | 0 | PARTIAL |
| E27_rule_transfer | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923650 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923650 | MODE_2_DYNAMIC_FIELD | 0.835 | 0.835 | 0.277 | 0 | PARTIAL |
| E18B_episodic_cdt_consolidation | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.355 | 0.355 | 0.924 | 0 | NEGATIVE |
| E18C_cdt_vs_no_cdt | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.676 | 0.676 | 0.713 | 0 | NEGATIVE |
| E19_no_lookup_audit | 923650 | MODE_2_DYNAMIC_FIELD | 0.835 | 0.835 | 0.277 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923650 | MODE_2_DYNAMIC_FIELD | 0.533 | 0.533 | 0.963 | 0 | NEGATIVE |
| E21_abstract_rule_param | 923650 | MODE_2_DYNAMIC_FIELD | 0.933 | 0.933 | 0.209 | 0 | PARTIAL |
| E22_unobserved_composition | 923650 | MODE_2_DYNAMIC_FIELD | 0.333 | 0.333 | 0.314 | 0 | NEGATIVE |
| E23_long_horizon | 923650 | MODE_2_DYNAMIC_FIELD | 0.513 | 0.918 | 0.513 | 0 | PARTIAL |
| E24_static_vs_dynamic | 923650 | MODE_2_DYNAMIC_FIELD | 0.258 | 0.258 | 0.632 | 0 | NEGATIVE |
| E25_cdt_experience_not_answer | 923175 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.953 | 0.953 | -0.174 | 0 | STRONG_POSITIVE |
| E26_experience_ablation | 923172 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.741 | 0.741 | 0.894 | 0 | NEGATIVE |
| E27_rule_transfer | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923651 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923651 | MODE_2_DYNAMIC_FIELD | 0.623 | 0.623 | -0.052 | 0 | NULL |
| E18B_episodic_cdt_consolidation | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.393 | 0.393 | 0.733 | 0 | NEGATIVE |
| E18C_cdt_vs_no_cdt | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.564 | 0.564 | 0.567 | 0 | NULL |
| E19_no_lookup_audit | 923651 | MODE_2_DYNAMIC_FIELD | 0.623 | 0.623 | -0.052 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923651 | MODE_2_DYNAMIC_FIELD | -0.825 | -0.825 | 0.799 | 0 | NEGATIVE |
| E21_abstract_rule_param | 923651 | MODE_2_DYNAMIC_FIELD | 0.040 | 0.040 | 0.130 | 0 | NULL |
| E22_unobserved_composition | 923651 | MODE_2_DYNAMIC_FIELD | 0.420 | 0.420 | 0.481 | 0 | NULL |
| E23_long_horizon | 923651 | MODE_2_DYNAMIC_FIELD | 0.840 | 0.853 | 0.840 | 0 | POSITIVE |
| E24_static_vs_dynamic | 923651 | MODE_2_DYNAMIC_FIELD | -0.448 | -0.448 | 0.960 | 0 | NEGATIVE |
| E25_cdt_experience_not_answer | 923174 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.283 | 0.283 | 0.899 | 0 | NEGATIVE |
| E26_experience_ablation | 923173 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | -0.129 | -0.129 | 0.684 | 0 | NEGATIVE |
| E27_rule_transfer | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923652 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923652 | MODE_2_DYNAMIC_FIELD | 0.931 | 0.931 | 0.324 | 0 | POSITIVE |
| E18B_episodic_cdt_consolidation | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.095 | 0.095 | 0.740 | 0 | NEGATIVE |
| E18C_cdt_vs_no_cdt | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.857 | 0.857 | 0.409 | 0 | POSITIVE |
| E19_no_lookup_audit | 923652 | MODE_2_DYNAMIC_FIELD | 0.931 | 0.931 | 0.324 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923652 | MODE_2_DYNAMIC_FIELD | -0.643 | -0.643 | 0.870 | 0 | NEGATIVE |
| E21_abstract_rule_param | 923652 | MODE_2_DYNAMIC_FIELD | 0.709 | 0.709 | 0.060 | 0 | PARTIAL |
| E22_unobserved_composition | 923652 | MODE_2_DYNAMIC_FIELD | 0.727 | 0.727 | 0.743 | 0 | PARTIAL |
| E23_long_horizon | 923652 | MODE_2_DYNAMIC_FIELD | 0.296 | 0.706 | 0.296 | 0 | PARTIAL |
| E24_static_vs_dynamic | 923652 | MODE_2_DYNAMIC_FIELD | 0.835 | 0.835 | 0.564 | 0 | POSITIVE |
| E25_cdt_experience_not_answer | 923169 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | -0.782 | -0.782 | 0.658 | 0 | NEGATIVE |
| E26_experience_ablation | 923170 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.435 | 0.435 | 0.148 | 0 | POSITIVE |
| E27_rule_transfer | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923653 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923653 | MODE_2_DYNAMIC_FIELD | 0.440 | 0.440 | 0.318 | 0 | NULL |
| E18B_episodic_cdt_consolidation | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.661 | 0.661 | 0.941 | 0 | NEGATIVE |
| E18C_cdt_vs_no_cdt | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.830 | 0.830 | 0.795 | 0 | PARTIAL |
| E19_no_lookup_audit | 923653 | MODE_2_DYNAMIC_FIELD | 0.440 | 0.440 | 0.318 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923653 | MODE_2_DYNAMIC_FIELD | 0.765 | 0.765 | 0.936 | 0 | NEGATIVE |
| E21_abstract_rule_param | 923653 | MODE_2_DYNAMIC_FIELD | 0.166 | 0.166 | 0.130 | 0 | NULL |
| E22_unobserved_composition | 923653 | MODE_2_DYNAMIC_FIELD | 0.684 | 0.684 | -0.638 | 0 | PARTIAL |
| E23_long_horizon | 923653 | MODE_2_DYNAMIC_FIELD | 0.231 | 0.934 | 0.231 | 0 | PARTIAL |
| E24_static_vs_dynamic | 923653 | MODE_2_DYNAMIC_FIELD | 0.744 | 0.744 | 0.883 | 0 | NEGATIVE |
| E25_cdt_experience_not_answer | 923168 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | -0.159 | -0.159 | -0.011 | 0 | NEGATIVE |
| E26_experience_ablation | 923171 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.592 | 0.592 | 0.196 | 0 | PARTIAL |
| E27_rule_transfer | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923654 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923654 | MODE_2_DYNAMIC_FIELD | -0.035 | -0.035 | 0.472 | 0 | NEGATIVE |
| E18B_episodic_cdt_consolidation | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.292 | 0.292 | 0.056 | 0 | NULL |
| E18C_cdt_vs_no_cdt | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.308 | 0.308 | 0.148 | 0 | PARTIAL |
| E19_no_lookup_audit | 923654 | MODE_2_DYNAMIC_FIELD | -0.035 | -0.035 | 0.472 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923654 | MODE_2_DYNAMIC_FIELD | 0.797 | 0.797 | 0.591 | 0 | PARTIAL |
| E21_abstract_rule_param | 923654 | MODE_2_DYNAMIC_FIELD | 0.235 | 0.235 | 0.449 | 0 | NULL |
| E22_unobserved_composition | 923654 | MODE_2_DYNAMIC_FIELD | 0.572 | 0.572 | 0.477 | 0 | PARTIAL |
| E23_long_horizon | 923654 | MODE_2_DYNAMIC_FIELD | -0.301 | 0.760 | -0.301 | 0 | PARTIAL |
| E24_static_vs_dynamic | 923654 | MODE_2_DYNAMIC_FIELD | -0.598 | -0.598 | 0.739 | 0 | NEGATIVE |
| E25_cdt_experience_not_answer | 923171 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.188 | 0.188 | 0.705 | 0 | NEGATIVE |
| E26_experience_ablation | 923168 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.210 | 0.210 | 0.883 | 0 | NEGATIVE |
| E27_rule_transfer | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923655 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923655 | MODE_2_DYNAMIC_FIELD | 0.968 | 0.968 | 0.653 | 0 | POSITIVE |
| E18B_episodic_cdt_consolidation | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.931 | 0.931 | 0.167 | 0 | STRONG_POSITIVE |
| E18C_cdt_vs_no_cdt | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.960 | 0.960 | 0.906 | 0 | PARTIAL |
| E19_no_lookup_audit | 923655 | MODE_2_DYNAMIC_FIELD | 0.968 | 0.968 | 0.653 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923655 | MODE_2_DYNAMIC_FIELD | 0.418 | 0.418 | 0.995 | 0 | NEGATIVE |
| E21_abstract_rule_param | 923655 | MODE_2_DYNAMIC_FIELD | 0.098 | 0.098 | 0.218 | 0 | NULL |
| E22_unobserved_composition | 923655 | MODE_2_DYNAMIC_FIELD | 0.604 | 0.604 | 0.640 | 0 | NULL |
| E23_long_horizon | 923655 | MODE_2_DYNAMIC_FIELD | 0.368 | 0.975 | 0.368 | 0 | PARTIAL |
| E24_static_vs_dynamic | 923655 | MODE_2_DYNAMIC_FIELD | -0.151 | -0.151 | 0.774 | 0 | NEGATIVE |
| E25_cdt_experience_not_answer | 923170 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.733 | 0.733 | 0.321 | 0 | PARTIAL |
| E26_experience_ablation | 923169 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.591 | 0.591 | 0.765 | 0 | NEGATIVE |
| E27_rule_transfer | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
