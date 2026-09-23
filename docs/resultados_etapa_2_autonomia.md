# Resultados — Etapa 2 autonomía de campo

Protocolo: `docs/etapa_2_autonomia_campo_experimentos.md`
Módulo: `src/field_autonomy_stage2.rs`
Rama: `exp/field-autonomy-stage-2`
Seeds desarrollo: 8 × `0xE1800..0xE1807`
Periferia: **numeric field-only** (GGUF presente en disco pero **no** usado en el núcleo de reglas).

## Hardening pass (este commit)

Nota: **hardening pass: more train + non-contrastive A/B + dyn-heavy**.

Cambios principales:
- `train_encoder_dynamics`: ya **no** usa B como negativo de A; encoder ligero = consistencia de punto cercano (negatives vacíos).
- Entrenamiento **dynamics-heavy**: 2–3 `train_transition` por par/época; fases enc-warm + dyn.
- Helper `augment_pairs(rule, pairs, n_copies, noise, seed)` usado en E18–E24.
- Más datos/épocas: E18A/B ≥16+4, aug×4, 100+400; E20 ≥32 pares + traj×8, 300; E21 ≥24 + multi-dx con `point_features_with_action`; E23 unroll 2–4; E24 ≥24/6, 400.
- E19 leakage sigue en 0. E27–E30 siguen `INVALID`.
- Umbrales honestos sin forzar PASS.

## Implementado vs diferido

| Estado | Experimentos |
|--------|--------------|
| Implementado (Fase A+B + parcial C/D) | FASE_A_field_only_scaffold, E18A_direct_no_cdt, E18B_episodic_cdt_consolidation, E18C_cdt_vs_no_cdt, E19_no_lookup_audit, E20_trajectory_vs_rule, E21_abstract_rule_param, E22_unobserved_composition, E23_long_horizon, E24_static_vs_dynamic, E25_cdt_experience_not_answer, E26_experience_ablation |
| Diferido (scaffold INVALID) | E27_rule_transfer, E28_continual_forgetting, E29_causal_intervention, E30_persistence_restart |

## Resumen before → after (8 seeds)

| Experimento | before hist | after hist | mean dyn before→after | mean static after | leak |
|-------------|-------------|------------|----------------------:|------------------:|-----:|
| FASE_A_field_only_scaffold | POSITIVE:8 | POSITIVE:8 | 0.000→0.000 | 0.000 | 0.0 |
| E18A_direct_no_cdt | NEGATIVE:2, NULL:2, PARTIAL:1, POSITIVE:2, STRONG_POSITIVE:1 | POSITIVE:4, STRONG_POSITIVE:2, NEGATIVE:1, PARTIAL:1 | 0.546→0.926 | 0.864 | 0.0 |
| E18B_episodic_cdt_consolidation | NEGATIVE:4, NULL:1, POSITIVE:2, STRONG_POSITIVE:1 | STRONG_POSITIVE:6, PARTIAL:1, POSITIVE:1 | 0.563→0.969 | 0.860 | 0.0 |
| E18C_cdt_vs_no_cdt | NEGATIVE:1, NULL:1, PARTIAL:3, POSITIVE:3 | PARTIAL:7, POSITIVE:1 | 0.626→0.979 | 0.634 | 0.0 |
| E19_no_lookup_audit | POSITIVE:8 | POSITIVE:8 | 0.546→0.926 | 0.864 | 0.0 |
| E20_trajectory_vs_rule | NEGATIVE:6, PARTIAL:2 | NULL:3, POSITIVE:3, PARTIAL:2 | 0.294→0.997 | 0.890 | 0.0 |
| E21_abstract_rule_param | NULL:5, PARTIAL:3 | POSITIVE:8 | 0.393→0.987 | 0.837 | 0.0 |
| E22_unobserved_composition | NEGATIVE:1, NULL:4, PARTIAL:3 | PARTIAL:7, NULL:1 | 0.533→0.881 | 0.901 | 0.0 |
| E23_long_horizon | NULL:1, PARTIAL:6, POSITIVE:1 | POSITIVE:7, PARTIAL:1 | 0.829→0.999 | 0.872 | 0.0 |
| E24_static_vs_dynamic | NEGATIVE:7, POSITIVE:1 | PARTIAL:7, POSITIVE:1 | 0.133→0.998 | 0.799 | 0.0 |
| E25_cdt_experience_not_answer | NEGATIVE:5, NULL:1, PARTIAL:1, STRONG_POSITIVE:1 | STRONG_POSITIVE:6, PARTIAL:1, POSITIVE:1 | 0.235→0.984 | 0.829 | 0.0 |
| E26_experience_ablation | NEGATIVE:4, PARTIAL:2, POSITIVE:2 | PARTIAL:7, POSITIVE:1 | 0.536→0.977 | 0.713 | 0.0 |
| E27_rule_transfer | INVALID:8 | INVALID:8 | 0.000→0.000 | 0.000 | 0.0 |
| E28_continual_forgetting | INVALID:8 | INVALID:8 | 0.000→0.000 | 0.000 | 0.0 |
| E29_causal_intervention | INVALID:8 | INVALID:8 | 0.000→0.000 | 0.000 | 0.0 |
| E30_persistence_restart | INVALID:8 | INVALID:8 | 0.000→0.000 | 0.000 | 0.0 |

**Leakage total (suma scores suite): 0**

## Lectura honesta (post-hardening)

- **E18A/B**: big lift en mean dyn (~0.55→~0.95); aún 1 seed NEGATIVE en E18A (varianza residual).
- **E18C**: C2 alto; muchos PARTIAL (C2 no siempre domina C1/C3 con margen estricto).
- **E19**: audit POSITIVE ×8, leakage=0.
- **E20**: de NEGATIVE:6 a mix POSITIVE/PARTIAL/NULL — rule arm ya no pierde sistemáticamente, pero traj arm también mejora (menos delta).
- **E21**: POSITIVE:8 con conditioning `(dx,dy)`; extrap dx=3 condicionado (nunca entrenado).
- **E22**: sigue débil (PARTIAL/NULL) — composición unobserved sin action switch sigue difícil.
- **E23**: POSITIVE frecuente en h1; h8 mixto (1 seed débil).
- **E24**: de NEGATIVE:7 a PARTIAL:7 + POSITIVE:1 — dyn ~0.99 supera static frozen-init, pero umbral `dyn > static+0.1 ∧ > random` rara vez da POSITIVE cuando static init ya es alto.
- **E25/E26**: heredan trainers; mejora clara vs before.
- **E27–E30**: INVALID (diferidos).

## Qué sigue débil

- E22 composition (un solo D_φ sin switch de acción).
- E24 veredicto POSITIVE raro (static cosine alto en init).
- E20: cuando traj≈rule, veredicto NULL (fair).
- E18A: outlier seed NEGATIVE.

## Controles

Cada E18* reporta static / linear / NN / table donde aplica. Table en estados novel ≈0. NN es memoria, no regla.

## Tabla cruda

| Exp | seed | mode | rule_gen | dyn | static | leak | verdict |
|-----|-----:|------|---------:|----:|-------:|-----:|---------|
| FASE_A_field_only_scaffold | 923648 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923648 | MODE_2_DYNAMIC_FIELD | 0.981 | 0.981 | 0.805 | 0 | STRONG_POSITIVE |
| E18B_episodic_cdt_consolidation | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.996 | 0.996 | 0.920 | 0 | POSITIVE |
| E18C_cdt_vs_no_cdt | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.965 | 0.965 | 0.255 | 0 | POSITIVE |
| E19_no_lookup_audit | 923648 | MODE_2_DYNAMIC_FIELD | 0.981 | 0.981 | 0.805 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923648 | MODE_2_DYNAMIC_FIELD | 0.999 | 0.999 | 0.772 | 0 | POSITIVE |
| E21_abstract_rule_param | 923648 | MODE_2_DYNAMIC_FIELD | 0.991 | 0.991 | 0.921 | 0 | POSITIVE |
| E22_unobserved_composition | 923648 | MODE_2_DYNAMIC_FIELD | 0.895 | 0.895 | 0.908 | 0 | PARTIAL |
| E23_long_horizon | 923648 | MODE_2_DYNAMIC_FIELD | 0.995 | 1.000 | 0.995 | 0 | POSITIVE |
| E24_static_vs_dynamic | 923648 | MODE_2_DYNAMIC_FIELD | 0.999 | 0.999 | 0.814 | 0 | PARTIAL |
| E25_cdt_experience_not_answer | 923173 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.988 | 0.988 | 0.616 | 0 | STRONG_POSITIVE |
| E26_experience_ablation | 923174 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.987 | 0.987 | 0.613 | 0 | PARTIAL |
| E27_rule_transfer | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923648 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923649 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923649 | MODE_2_DYNAMIC_FIELD | 0.970 | 0.970 | 0.897 | 0 | POSITIVE |
| E18B_episodic_cdt_consolidation | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.934 | 0.934 | 0.829 | 0 | STRONG_POSITIVE |
| E18C_cdt_vs_no_cdt | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.981 | 0.981 | 0.498 | 0 | PARTIAL |
| E19_no_lookup_audit | 923649 | MODE_2_DYNAMIC_FIELD | 0.970 | 0.970 | 0.897 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923649 | MODE_2_DYNAMIC_FIELD | 1.000 | 1.000 | 0.966 | 0 | NULL |
| E21_abstract_rule_param | 923649 | MODE_2_DYNAMIC_FIELD | 0.949 | 0.949 | 0.830 | 0 | POSITIVE |
| E22_unobserved_composition | 923649 | MODE_2_DYNAMIC_FIELD | 0.910 | 0.910 | 0.916 | 0 | PARTIAL |
| E23_long_horizon | 923649 | MODE_2_DYNAMIC_FIELD | 0.998 | 0.997 | 0.998 | 0 | POSITIVE |
| E24_static_vs_dynamic | 923649 | MODE_2_DYNAMIC_FIELD | 0.997 | 0.997 | 0.718 | 0 | POSITIVE |
| E25_cdt_experience_not_answer | 923172 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.974 | 0.974 | 0.790 | 0 | STRONG_POSITIVE |
| E26_experience_ablation | 923175 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.990 | 0.990 | 0.800 | 0 | PARTIAL |
| E27_rule_transfer | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923649 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923650 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923650 | MODE_2_DYNAMIC_FIELD | 0.986 | 0.986 | 0.934 | 0 | POSITIVE |
| E18B_episodic_cdt_consolidation | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.985 | 0.985 | 0.875 | 0 | STRONG_POSITIVE |
| E18C_cdt_vs_no_cdt | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.978 | 0.978 | 0.843 | 0 | PARTIAL |
| E19_no_lookup_audit | 923650 | MODE_2_DYNAMIC_FIELD | 0.986 | 0.986 | 0.934 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923650 | MODE_2_DYNAMIC_FIELD | 0.986 | 0.986 | 0.845 | 0 | POSITIVE |
| E21_abstract_rule_param | 923650 | MODE_2_DYNAMIC_FIELD | 0.988 | 0.988 | 0.875 | 0 | POSITIVE |
| E22_unobserved_composition | 923650 | MODE_2_DYNAMIC_FIELD | 0.912 | 0.912 | 0.937 | 0 | PARTIAL |
| E23_long_horizon | 923650 | MODE_2_DYNAMIC_FIELD | 0.998 | 0.999 | 0.998 | 0 | POSITIVE |
| E24_static_vs_dynamic | 923650 | MODE_2_DYNAMIC_FIELD | 0.998 | 0.998 | 0.789 | 0 | PARTIAL |
| E25_cdt_experience_not_answer | 923175 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.960 | 0.960 | 0.851 | 0 | STRONG_POSITIVE |
| E26_experience_ablation | 923172 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.989 | 0.989 | 0.792 | 0 | PARTIAL |
| E27_rule_transfer | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923650 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923651 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923651 | MODE_2_DYNAMIC_FIELD | 0.986 | 0.986 | 0.784 | 0 | STRONG_POSITIVE |
| E18B_episodic_cdt_consolidation | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.950 | 0.950 | 0.864 | 0 | STRONG_POSITIVE |
| E18C_cdt_vs_no_cdt | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.986 | 0.986 | 0.441 | 0 | PARTIAL |
| E19_no_lookup_audit | 923651 | MODE_2_DYNAMIC_FIELD | 0.986 | 0.986 | 0.784 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923651 | MODE_2_DYNAMIC_FIELD | 1.000 | 1.000 | 0.927 | 0 | PARTIAL |
| E21_abstract_rule_param | 923651 | MODE_2_DYNAMIC_FIELD | 0.993 | 0.993 | 0.837 | 0 | POSITIVE |
| E22_unobserved_composition | 923651 | MODE_2_DYNAMIC_FIELD | 0.915 | 0.915 | 0.918 | 0 | PARTIAL |
| E23_long_horizon | 923651 | MODE_2_DYNAMIC_FIELD | 0.930 | 0.999 | 0.930 | 0 | POSITIVE |
| E24_static_vs_dynamic | 923651 | MODE_2_DYNAMIC_FIELD | 0.998 | 0.998 | 0.875 | 0 | PARTIAL |
| E25_cdt_experience_not_answer | 923174 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.974 | 0.974 | 0.741 | 0 | STRONG_POSITIVE |
| E26_experience_ablation | 923173 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.993 | 0.993 | 0.758 | 0 | PARTIAL |
| E27_rule_transfer | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923651 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923652 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923652 | MODE_2_DYNAMIC_FIELD | 0.931 | 0.931 | 0.885 | 0 | PARTIAL |
| E18B_episodic_cdt_consolidation | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.935 | 0.935 | 0.835 | 0 | STRONG_POSITIVE |
| E18C_cdt_vs_no_cdt | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.984 | 0.984 | 0.829 | 0 | PARTIAL |
| E19_no_lookup_audit | 923652 | MODE_2_DYNAMIC_FIELD | 0.931 | 0.931 | 0.885 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923652 | MODE_2_DYNAMIC_FIELD | 0.999 | 0.999 | 0.707 | 0 | POSITIVE |
| E21_abstract_rule_param | 923652 | MODE_2_DYNAMIC_FIELD | 0.989 | 0.989 | 0.728 | 0 | POSITIVE |
| E22_unobserved_composition | 923652 | MODE_2_DYNAMIC_FIELD | 0.609 | 0.609 | 0.754 | 0 | NULL |
| E23_long_horizon | 923652 | MODE_2_DYNAMIC_FIELD | 0.126 | 1.000 | 0.126 | 0 | PARTIAL |
| E24_static_vs_dynamic | 923652 | MODE_2_DYNAMIC_FIELD | 0.996 | 0.996 | 0.778 | 0 | PARTIAL |
| E25_cdt_experience_not_answer | 923169 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.994 | 0.994 | 0.949 | 0 | PARTIAL |
| E26_experience_ablation | 923170 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.966 | 0.966 | 0.757 | 0 | PARTIAL |
| E27_rule_transfer | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923652 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923653 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923653 | MODE_2_DYNAMIC_FIELD | 0.569 | 0.569 | 0.774 | 0 | NEGATIVE |
| E18B_episodic_cdt_consolidation | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.977 | 0.977 | 0.930 | 0 | PARTIAL |
| E18C_cdt_vs_no_cdt | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.982 | 0.982 | 0.637 | 0 | PARTIAL |
| E19_no_lookup_audit | 923653 | MODE_2_DYNAMIC_FIELD | 0.569 | 0.569 | 0.774 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923653 | MODE_2_DYNAMIC_FIELD | 0.996 | 0.996 | 0.996 | 0 | NULL |
| E21_abstract_rule_param | 923653 | MODE_2_DYNAMIC_FIELD | 0.989 | 0.989 | 0.879 | 0 | POSITIVE |
| E22_unobserved_composition | 923653 | MODE_2_DYNAMIC_FIELD | 0.936 | 0.936 | 0.923 | 0 | PARTIAL |
| E23_long_horizon | 923653 | MODE_2_DYNAMIC_FIELD | 0.951 | 0.999 | 0.951 | 0 | POSITIVE |
| E24_static_vs_dynamic | 923653 | MODE_2_DYNAMIC_FIELD | 0.999 | 0.999 | 0.761 | 0 | PARTIAL |
| E25_cdt_experience_not_answer | 923168 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.995 | 0.995 | 0.859 | 0 | STRONG_POSITIVE |
| E26_experience_ablation | 923171 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.996 | 0.996 | 0.666 | 0 | PARTIAL |
| E27_rule_transfer | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923653 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923654 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923654 | MODE_2_DYNAMIC_FIELD | 0.996 | 0.996 | 0.921 | 0 | POSITIVE |
| E18B_episodic_cdt_consolidation | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.988 | 0.988 | 0.799 | 0 | STRONG_POSITIVE |
| E18C_cdt_vs_no_cdt | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.974 | 0.974 | 0.776 | 0 | PARTIAL |
| E19_no_lookup_audit | 923654 | MODE_2_DYNAMIC_FIELD | 0.996 | 0.996 | 0.921 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923654 | MODE_2_DYNAMIC_FIELD | 1.000 | 1.000 | 0.919 | 0 | PARTIAL |
| E21_abstract_rule_param | 923654 | MODE_2_DYNAMIC_FIELD | 0.996 | 0.996 | 0.734 | 0 | POSITIVE |
| E22_unobserved_composition | 923654 | MODE_2_DYNAMIC_FIELD | 0.937 | 0.937 | 0.907 | 0 | PARTIAL |
| E23_long_horizon | 923654 | MODE_2_DYNAMIC_FIELD | 0.984 | 1.000 | 0.984 | 0 | POSITIVE |
| E24_static_vs_dynamic | 923654 | MODE_2_DYNAMIC_FIELD | 0.998 | 0.998 | 0.779 | 0 | PARTIAL |
| E25_cdt_experience_not_answer | 923171 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.994 | 0.994 | 0.931 | 0 | POSITIVE |
| E26_experience_ablation | 923168 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.967 | 0.967 | 0.639 | 0 | PARTIAL |
| E27_rule_transfer | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923654 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| FASE_A_field_only_scaffold | 923655 | MODE_2_DYNAMIC_FIELD | 0.000 | 0.000 | 0.000 | 0 | POSITIVE |
| E18A_direct_no_cdt | 923655 | MODE_2_DYNAMIC_FIELD | 0.987 | 0.987 | 0.915 | 0 | POSITIVE |
| E18B_episodic_cdt_consolidation | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.986 | 0.986 | 0.826 | 0 | STRONG_POSITIVE |
| E18C_cdt_vs_no_cdt | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.985 | 0.985 | 0.795 | 0 | PARTIAL |
| E19_no_lookup_audit | 923655 | MODE_2_DYNAMIC_FIELD | 0.987 | 0.987 | 0.915 | 0 | POSITIVE |
| E20_trajectory_vs_rule | 923655 | MODE_2_DYNAMIC_FIELD | 0.999 | 0.999 | 0.985 | 0 | NULL |
| E21_abstract_rule_param | 923655 | MODE_2_DYNAMIC_FIELD | 1.000 | 1.000 | 0.894 | 0 | POSITIVE |
| E22_unobserved_composition | 923655 | MODE_2_DYNAMIC_FIELD | 0.937 | 0.937 | 0.946 | 0 | PARTIAL |
| E23_long_horizon | 923655 | MODE_2_DYNAMIC_FIELD | 0.998 | 1.000 | 0.998 | 0 | POSITIVE |
| E24_static_vs_dynamic | 923655 | MODE_2_DYNAMIC_FIELD | 0.998 | 0.998 | 0.876 | 0 | PARTIAL |
| E25_cdt_experience_not_answer | 923170 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.991 | 0.991 | 0.894 | 0 | STRONG_POSITIVE |
| E26_experience_ablation | 923169 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.928 | 0.928 | 0.679 | 0 | POSITIVE |
| E27_rule_transfer | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E28_continual_forgetting | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E29_causal_intervention | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |
| E30_persistence_restart | 923655 | MODE_3_DYNAMIC_PLUS_CDT_TRAINING | 0.000 | 0.000 | 0.000 | 0 | INVALID |

## Suite summary (verdict counts by experiment)

| Experiment | verdict histogram |
|------------|-------------------|
| E18A_direct_no_cdt | NEGATIVE:1, PARTIAL:1, POSITIVE:4, STRONG_POSITIVE:2 |
| E18B_episodic_cdt_consolidation | PARTIAL:1, POSITIVE:1, STRONG_POSITIVE:6 |
| E18C_cdt_vs_no_cdt | PARTIAL:7, POSITIVE:1 |
| E19_no_lookup_audit | POSITIVE:8 |
| E20_trajectory_vs_rule | NULL:3, PARTIAL:2, POSITIVE:3 |
| E21_abstract_rule_param | POSITIVE:8 |
| E22_unobserved_composition | NULL:1, PARTIAL:7 |
| E23_long_horizon | PARTIAL:1, POSITIVE:7 |
| E24_static_vs_dynamic | PARTIAL:7, POSITIVE:1 |
| E25_cdt_experience_not_answer | PARTIAL:1, POSITIVE:1, STRONG_POSITIVE:6 |
| E26_experience_ablation | PARTIAL:7, POSITIVE:1 |
| E27_rule_transfer | INVALID:8 |
| E28_continual_forgetting | INVALID:8 |
| E29_causal_intervention | INVALID:8 |
| E30_persistence_restart | INVALID:8 |
| FASE_A_field_only_scaffold | POSITIVE:8 |

