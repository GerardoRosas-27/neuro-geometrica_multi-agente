# Resultados experimentos 11–17 — campo entrenable

Rama: `exp/liquid-inference-experiments-8-9-10`

Protocolo: `docs/experimentos_11_17_campo_entrenable.md`

Commit al correr: 26d587d.
Hardware: x86_64.
rustc: rustc 1.98.1 (48a229cea 2026-09-01).
Semillas: 8 (0xE1100..0xE1107). Escala N=8 cubierta en E11 (`accuracy_ood`).
Periferia: gemma-shaped-lexicon (GGUF absent; honest).
GGUF: ausente → E12 SKIPPED_NO_GGUF.

## Tabla de registro

# Registry — experimentos 11–17 (campo entrenable)

| Exp | seed | mode | acc_seen | acc_unseen | acc_ood | margin | knn/top1 | dR_A | leak_rqm | verdict |
|-----|-----:|------|---------:|-----------:|--------:|-------:|---------:|-----:|---------:|---------|
| E11_trainable_field_encoder | 921856 | STATIC_FIELD | 0.769 | 0.833 | 0.625 | 0.611 | 0.769 | 0.000 | false | PASS: trained encoder geometry > random projector |
| E12_crosslingual_concept | 921856 | STATIC_FIELD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | false | SKIPPED_NO_GGUF |
| E13_relational_field | 921856 | DYNAMIC_FIELD | 0.200 | 0.333 | 0.333 | -1.265 | 0.200 | 0.000 | false | PARTIAL: dynamic/static field generaliza holdout (RQM solo control) |
| E14_incremental_forgetting | 921856 | STATIC_FIELD | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.031 | false | PASS: incremental recall retained; adversarial retain>=0.5 |
| E15_semantic_without_labels | 921856 | STATIC_FIELD | 1.000 | 0.333 | 0.000 | -0.000 | 0.000 | 0.000 | false | PARTIAL: weak structural separation without class labels |
| E16_field_dynamics | 921856 | DYNAMIC_FIELD | 1.000 | 1.000 | 1.000 | 0.490 | 1.000 | 0.000 | true | PASS: Dphi predicts transitions; rollout partial; RQM control separate |
| E17_never_observed_states | 921856 | DYNAMIC_FIELD | 0.995 | 1.000 | 1.000 | 0.007 | 0.998 | 0.000 | true | PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) |
| E11_trainable_field_encoder | 921857 | STATIC_FIELD | 0.923 | 1.000 | 0.625 | 0.874 | 0.923 | 0.000 | false | PASS: trained encoder geometry > random projector |
| E12_crosslingual_concept | 921857 | STATIC_FIELD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | false | SKIPPED_NO_GGUF |
| E13_relational_field | 921857 | DYNAMIC_FIELD | 0.200 | 0.333 | 0.333 | -1.197 | 0.200 | 0.000 | false | PARTIAL: dynamic/static field generaliza holdout (RQM solo control) |
| E14_incremental_forgetting | 921857 | STATIC_FIELD | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | -0.006 | false | PASS: incremental recall retained; adversarial retain>=0.5 |
| E15_semantic_without_labels | 921857 | STATIC_FIELD | 1.000 | 0.333 | 0.000 | -0.001 | 0.000 | 0.000 | false | PARTIAL: weak structural separation without class labels |
| E16_field_dynamics | 921857 | DYNAMIC_FIELD | 1.000 | 1.000 | 1.000 | 0.013 | 0.985 | 0.000 | true | WEAK: mild dynamics; may not beat static/NN |
| E17_never_observed_states | 921857 | DYNAMIC_FIELD | 0.995 | 1.000 | 1.000 | 0.009 | 0.999 | 0.000 | true | PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) |
| E11_trainable_field_encoder | 921858 | STATIC_FIELD | 0.769 | 1.000 | 0.375 | 0.714 | 0.769 | 0.000 | false | PASS: trained encoder geometry > random projector |
| E12_crosslingual_concept | 921858 | STATIC_FIELD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | false | SKIPPED_NO_GGUF |
| E13_relational_field | 921858 | DYNAMIC_FIELD | 0.200 | 0.333 | 0.333 | -1.113 | 0.200 | 0.000 | false | PARTIAL: dynamic/static field generaliza holdout (RQM solo control) |
| E14_incremental_forgetting | 921858 | STATIC_FIELD | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | -0.005 | false | PASS: incremental recall retained; adversarial retain>=0.5 |
| E15_semantic_without_labels | 921858 | STATIC_FIELD | 1.000 | 0.333 | 0.000 | -0.000 | 0.000 | 0.000 | false | PARTIAL: weak structural separation without class labels |
| E16_field_dynamics | 921858 | DYNAMIC_FIELD | 1.000 | 1.000 | 1.000 | 0.547 | 1.000 | 0.000 | true | PASS: Dphi predicts transitions; rollout partial; RQM control separate |
| E17_never_observed_states | 921858 | DYNAMIC_FIELD | 0.995 | 1.000 | 1.000 | 0.011 | 0.998 | 0.000 | true | PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) |
| E11_trainable_field_encoder | 921859 | STATIC_FIELD | 0.846 | 0.833 | 0.750 | 0.914 | 0.846 | 0.000 | false | PASS: trained encoder geometry > random projector |
| E12_crosslingual_concept | 921859 | STATIC_FIELD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | false | SKIPPED_NO_GGUF |
| E13_relational_field | 921859 | DYNAMIC_FIELD | 0.400 | 0.333 | 0.333 | -1.187 | 0.400 | 0.000 | false | PARTIAL: dynamic/static field generaliza holdout (RQM solo control) |
| E14_incremental_forgetting | 921859 | STATIC_FIELD | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.013 | false | PASS: incremental recall retained; adversarial retain>=0.5 |
| E15_semantic_without_labels | 921859 | STATIC_FIELD | 1.000 | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 | false | FAIL: no structural signal for unseen incompatible combos |
| E16_field_dynamics | 921859 | DYNAMIC_FIELD | 1.000 | 1.000 | 1.000 | 0.528 | 1.000 | 0.000 | true | PASS: Dphi predicts transitions; rollout partial; RQM control separate |
| E17_never_observed_states | 921859 | DYNAMIC_FIELD | 0.995 | 1.000 | 1.000 | 0.009 | 0.999 | 0.000 | true | PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) |
| E11_trainable_field_encoder | 921860 | STATIC_FIELD | 0.692 | 0.500 | 0.500 | 0.167 | 0.692 | 0.000 | false | PARTIAL: geometry improved but not all criteria |
| E12_crosslingual_concept | 921860 | STATIC_FIELD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | false | SKIPPED_NO_GGUF |
| E13_relational_field | 921860 | DYNAMIC_FIELD | 0.400 | 0.333 | 0.333 | -1.094 | 0.400 | 0.000 | false | PARTIAL: dynamic/static field generaliza holdout (RQM solo control) |
| E14_incremental_forgetting | 921860 | STATIC_FIELD | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | -0.004 | false | PASS: incremental recall retained; adversarial retain>=0.5 |
| E15_semantic_without_labels | 921860 | STATIC_FIELD | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | false | FAIL: no structural signal for unseen incompatible combos |
| E16_field_dynamics | 921860 | DYNAMIC_FIELD | 1.000 | 1.000 | 1.000 | 0.814 | 1.000 | 0.000 | true | PASS: Dphi predicts transitions; rollout partial; RQM control separate |
| E17_never_observed_states | 921860 | DYNAMIC_FIELD | 0.996 | 1.000 | 1.000 | 0.019 | 0.999 | 0.000 | true | PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) |
| E11_trainable_field_encoder | 921861 | STATIC_FIELD | 1.000 | 1.000 | 0.750 | 0.866 | 1.000 | 0.000 | false | PASS: trained encoder geometry > random projector |
| E12_crosslingual_concept | 921861 | STATIC_FIELD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | false | SKIPPED_NO_GGUF |
| E13_relational_field | 921861 | DYNAMIC_FIELD | 0.600 | 0.333 | 0.333 | -1.195 | 0.600 | 0.000 | false | PARTIAL: dynamic/static field generaliza holdout (RQM solo control) |
| E14_incremental_forgetting | 921861 | STATIC_FIELD | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | -0.003 | false | PASS: incremental recall retained; adversarial retain>=0.5 |
| E15_semantic_without_labels | 921861 | STATIC_FIELD | 1.000 | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 | false | FAIL: no structural signal for unseen incompatible combos |
| E16_field_dynamics | 921861 | DYNAMIC_FIELD | 0.411 | 1.000 | 1.000 | -0.580 | 0.411 | 0.000 | true | WEAK: mild dynamics; may not beat static/NN |
| E17_never_observed_states | 921861 | DYNAMIC_FIELD | 0.996 | 1.000 | 1.000 | 0.012 | 0.999 | 0.000 | true | PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) |
| E11_trainable_field_encoder | 921862 | STATIC_FIELD | 1.000 | 0.833 | 0.750 | 0.936 | 1.000 | 0.000 | false | PASS: trained encoder geometry > random projector |
| E12_crosslingual_concept | 921862 | STATIC_FIELD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | false | SKIPPED_NO_GGUF |
| E13_relational_field | 921862 | DYNAMIC_FIELD | 0.200 | 0.333 | 0.333 | -1.309 | 0.200 | 0.000 | false | PARTIAL: dynamic/static field generaliza holdout (RQM solo control) |
| E14_incremental_forgetting | 921862 | STATIC_FIELD | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.013 | false | PASS: incremental recall retained; adversarial retain>=0.5 |
| E15_semantic_without_labels | 921862 | STATIC_FIELD | 1.000 | 0.333 | 0.000 | -0.000 | 0.000 | 0.000 | false | PARTIAL: weak structural separation without class labels |
| E16_field_dynamics | 921862 | DYNAMIC_FIELD | 1.000 | 1.000 | 1.000 | 0.616 | 1.000 | 0.000 | true | PASS: Dphi predicts transitions; rollout partial; RQM control separate |
| E17_never_observed_states | 921862 | DYNAMIC_FIELD | 0.995 | 1.000 | 1.000 | 0.010 | 0.998 | 0.000 | true | PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) |
| E11_trainable_field_encoder | 921863 | STATIC_FIELD | 0.692 | 0.833 | 0.750 | 0.857 | 0.692 | 0.000 | false | PARTIAL: geometry improved but not all criteria |
| E12_crosslingual_concept | 921863 | STATIC_FIELD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | false | SKIPPED_NO_GGUF |
| E13_relational_field | 921863 | DYNAMIC_FIELD | 0.200 | 0.333 | 0.333 | -1.183 | 0.200 | 0.000 | false | PARTIAL: dynamic/static field generaliza holdout (RQM solo control) |
| E14_incremental_forgetting | 921863 | STATIC_FIELD | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.033 | false | PASS: incremental recall retained; adversarial retain>=0.5 |
| E15_semantic_without_labels | 921863 | STATIC_FIELD | 1.000 | 0.333 | 0.000 | -0.000 | 0.000 | 0.000 | false | PARTIAL: weak structural separation without class labels |
| E16_field_dynamics | 921863 | DYNAMIC_FIELD | 1.000 | 1.000 | 1.000 | 0.837 | 1.000 | 0.000 | true | PASS: Dphi predicts transitions; rollout partial; RQM control separate |
| E17_never_observed_states | 921863 | DYNAMIC_FIELD | 0.996 | 1.000 | 1.000 | 0.007 | 0.998 | 0.000 | true | PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) |


## Notas por corrida

- E11_trainable_field_encoder seed=921856 verdict=PASS: trained encoder geometry > random projector notes=raw_margin=0.000 rand_margin=0.013 pre_margin=0.062 post_margin=0.611 raw_knn=0.538 rand_knn=0.308 knn=0.769 para=0.833 sil=0.667 stab=0.991 causal_delta=0.775 loss_end=1.9819 elapsed_ms=43 raw_intra=1.000 raw_inter=1.000
- E12_crosslingual_concept seed=921856 verdict=SKIPPED_NO_GGUF notes=GEMMA2_GGUF unset y no hay .gguf bajo models/; harness exige FrozenGemma2Probe real. NO se afirma cross-lingual con léxico. Tests compilados; ejecución omitida honestamente.
- E13_relational_field seed=921856 verdict=PARTIAL: dynamic/static field generaliza holdout (RQM solo control) notes=A_table seen=1.00 un=0.00; B_RQM seen=0.60 un=0.00; C_static seen=0.20 un=0.33; D_dynamic seen=0.20 un=0.33
- E14_incremental_forgetting seed=921856 verdict=PASS: incremental recall retained; adversarial retain>=0.5 notes=dR_A=0.031 dR_B=-0.000 dR_C=-0.002 final_acc=1.000 adv_retain_ABC=1.000 curve_A=Some([0.9636326945073439, 0.9994144162560841, 0.9986767511792158, 0.9950265375153415]) curve_B=Some([0.9999999999999987, 0.9994909068894604, 0.9999996783080494]) local_bias=1
- E15_semantic_without_labels seed=921856 verdict=PARTIAL: weak structural separation without class labels notes=E_ok=54.8610 E_bad=55.0063 d_man_ok=0.0014 d_man_bad=0.0012 stab_ok=0.9675 stab_bad=0.9675 structure=1/3
- E16_field_dynamics seed=921856 verdict=PASS: Dphi predicts transitions; rollout partial; RQM control separate notes=table_ok=true nn_next=B rqm_direct=1 rqm_compose=3 cdt_pred=1 cos_B=1.000 rollB=1.000 rollC=1.000 rollD=1.000 static_cosAB=0.499 causal_pre=0.510 audit_clean_for_AtoD_train=true
- E17_never_observed_states seed=921856 verdict=PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) notes=dist_C=0.0019 dist_B=0.0093 comp_dist=0.0000 rqm_direct_edge=false rqm_compose_pred=2 audit_train=false audit_rqm_compose=true audit_bank=false valid_unseen=true
- E11_trainable_field_encoder seed=921857 verdict=PASS: trained encoder geometry > random projector notes=raw_margin=0.000 rand_margin=0.005 pre_margin=0.008 post_margin=0.874 raw_knn=0.231 rand_knn=0.231 knn=0.923 para=1.000 sil=0.815 stab=0.990 causal_delta=0.997 loss_end=1.4388 elapsed_ms=33 raw_intra=1.000 raw_inter=1.000
- E12_crosslingual_concept seed=921857 verdict=SKIPPED_NO_GGUF notes=GEMMA2_GGUF unset y no hay .gguf bajo models/; harness exige FrozenGemma2Probe real. NO se afirma cross-lingual con léxico. Tests compilados; ejecución omitida honestamente.
- E13_relational_field seed=921857 verdict=PARTIAL: dynamic/static field generaliza holdout (RQM solo control) notes=A_table seen=1.00 un=0.00; B_RQM seen=0.60 un=0.00; C_static seen=0.20 un=0.33; D_dynamic seen=0.20 un=0.33
- E14_incremental_forgetting seed=921857 verdict=PASS: incremental recall retained; adversarial retain>=0.5 notes=dR_A=-0.006 dR_B=-0.000 dR_C=-0.001 final_acc=1.000 adv_retain_ABC=1.000 curve_A=Some([0.9993554520458104, 0.9992031755682205, 0.9994441837826883, 0.9932810177710603]) curve_B=Some([1.0000000000000004, 0.9994338473776411, 0.9999853519829455]) local_bias=1
- E15_semantic_without_labels seed=921857 verdict=PARTIAL: weak structural separation without class labels notes=E_ok=58.3383 E_bad=58.6038 d_man_ok=0.0023 d_man_bad=0.0016 stab_ok=0.9723 stab_bad=0.9723 structure=1/3
- E16_field_dynamics seed=921857 verdict=WEAK: mild dynamics; may not beat static/NN notes=table_ok=true nn_next=B rqm_direct=1 rqm_compose=3 cdt_pred=1 cos_B=0.985 rollB=0.985 rollC=0.863 rollD=0.995 static_cosAB=0.985 causal_pre=0.971 audit_clean_for_AtoD_train=true
- E17_never_observed_states seed=921857 verdict=PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) notes=dist_C=0.0010 dist_B=0.0103 comp_dist=-0.0000 rqm_direct_edge=false rqm_compose_pred=2 audit_train=false audit_rqm_compose=true audit_bank=false valid_unseen=true
- E11_trainable_field_encoder seed=921858 verdict=PASS: trained encoder geometry > random projector notes=raw_margin=0.000 rand_margin=0.004 pre_margin=0.002 post_margin=0.714 raw_knn=0.308 rand_knn=0.154 knn=0.769 para=1.000 sil=0.717 stab=0.991 causal_delta=0.370 loss_end=5.7307 elapsed_ms=30 raw_intra=1.000 raw_inter=1.000
- E12_crosslingual_concept seed=921858 verdict=SKIPPED_NO_GGUF notes=GEMMA2_GGUF unset y no hay .gguf bajo models/; harness exige FrozenGemma2Probe real. NO se afirma cross-lingual con léxico. Tests compilados; ejecución omitida honestamente.
- E13_relational_field seed=921858 verdict=PARTIAL: dynamic/static field generaliza holdout (RQM solo control) notes=A_table seen=1.00 un=0.00; B_RQM seen=0.60 un=0.00; C_static seen=0.40 un=0.33; D_dynamic seen=0.20 un=0.33
- E14_incremental_forgetting seed=921858 verdict=PASS: incremental recall retained; adversarial retain>=0.5 notes=dR_A=-0.005 dR_B=-0.000 dR_C=-0.002 final_acc=1.000 adv_retain_ABC=1.000 curve_A=Some([0.9993899415358866, 0.9986248593078386, 0.9991149750868686, 0.9939583602329681]) curve_B=Some([1.0, 0.9995320266976031, 0.9999983479999502]) local_bias=1
- E15_semantic_without_labels seed=921858 verdict=PARTIAL: weak structural separation without class labels notes=E_ok=62.7446 E_bad=62.8205 d_man_ok=0.0010 d_man_bad=0.0009 stab_ok=0.9697 stab_bad=0.9697 structure=1/3
- E16_field_dynamics seed=921858 verdict=PASS: Dphi predicts transitions; rollout partial; RQM control separate notes=table_ok=true nn_next=B rqm_direct=1 rqm_compose=3 cdt_pred=1 cos_B=1.000 rollB=1.000 rollC=1.000 rollD=1.000 static_cosAB=0.433 causal_pre=0.453 audit_clean_for_AtoD_train=true
- E17_never_observed_states seed=921858 verdict=PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) notes=dist_C=0.0020 dist_B=0.0125 comp_dist=-0.0000 rqm_direct_edge=false rqm_compose_pred=2 audit_train=false audit_rqm_compose=true audit_bank=false valid_unseen=true
- E11_trainable_field_encoder seed=921859 verdict=PASS: trained encoder geometry > random projector notes=raw_margin=0.000 rand_margin=0.033 pre_margin=0.015 post_margin=0.914 raw_knn=0.231 rand_knn=0.462 knn=0.846 para=0.833 sil=0.804 stab=0.991 causal_delta=0.763 loss_end=4.0637 elapsed_ms=30 raw_intra=1.000 raw_inter=1.000
- E12_crosslingual_concept seed=921859 verdict=SKIPPED_NO_GGUF notes=GEMMA2_GGUF unset y no hay .gguf bajo models/; harness exige FrozenGemma2Probe real. NO se afirma cross-lingual con léxico. Tests compilados; ejecución omitida honestamente.
- E13_relational_field seed=921859 verdict=PARTIAL: dynamic/static field generaliza holdout (RQM solo control) notes=A_table seen=1.00 un=0.00; B_RQM seen=0.60 un=0.00; C_static seen=0.20 un=0.33; D_dynamic seen=0.40 un=0.33
- E14_incremental_forgetting seed=921859 verdict=PASS: incremental recall retained; adversarial retain>=0.5 notes=dR_A=0.013 dR_B=-0.000 dR_C=-0.002 final_acc=1.000 adv_retain_ABC=1.000 curve_A=Some([0.9814242460200537, 0.9989953928525179, 0.9991079615927565, 0.9941477814830343]) curve_B=Some([1.0, 0.9997120106846641, 0.9999999259476052]) local_bias=1
- E15_semantic_without_labels seed=921859 verdict=FAIL: no structural signal for unseen incompatible combos notes=E_ok=64.3024 E_bad=64.2711 d_man_ok=0.0019 d_man_bad=0.0014 stab_ok=0.9742 stab_bad=0.9742 structure=0/3
- E16_field_dynamics seed=921859 verdict=PASS: Dphi predicts transitions; rollout partial; RQM control separate notes=table_ok=true nn_next=B rqm_direct=1 rqm_compose=3 cdt_pred=1 cos_B=1.000 rollB=1.000 rollC=1.000 rollD=1.000 static_cosAB=0.465 causal_pre=0.472 audit_clean_for_AtoD_train=true
- E17_never_observed_states seed=921859 verdict=PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) notes=dist_C=0.0014 dist_B=0.0099 comp_dist=0.0000 rqm_direct_edge=false rqm_compose_pred=2 audit_train=false audit_rqm_compose=true audit_bank=false valid_unseen=true
- E11_trainable_field_encoder seed=921860 verdict=PARTIAL: geometry improved but not all criteria notes=raw_margin=0.000 rand_margin=-0.005 pre_margin=0.015 post_margin=0.167 raw_knn=0.308 rand_knn=0.077 knn=0.692 para=0.500 sil=0.326 stab=0.991 causal_delta=0.646 loss_end=1.9021 elapsed_ms=33 raw_intra=1.000 raw_inter=1.000
- E12_crosslingual_concept seed=921860 verdict=SKIPPED_NO_GGUF notes=GEMMA2_GGUF unset y no hay .gguf bajo models/; harness exige FrozenGemma2Probe real. NO se afirma cross-lingual con léxico. Tests compilados; ejecución omitida honestamente.
- E13_relational_field seed=921860 verdict=PARTIAL: dynamic/static field generaliza holdout (RQM solo control) notes=A_table seen=1.00 un=0.00; B_RQM seen=0.60 un=0.00; C_static seen=0.40 un=0.33; D_dynamic seen=0.40 un=0.33
- E14_incremental_forgetting seed=921860 verdict=PASS: incremental recall retained; adversarial retain>=0.5 notes=dR_A=-0.004 dR_B=-0.002 dR_C=-0.000 final_acc=1.000 adv_retain_ABC=1.000 curve_A=Some([0.9993784765910171, 0.9989950657000826, 0.9991580608623022, 0.9957469488502625]) curve_B=Some([1.0, 0.9996357028666917, 0.9977503311921173]) local_bias=1
- E15_semantic_without_labels seed=921860 verdict=FAIL: no structural signal for unseen incompatible combos notes=E_ok=53.8238 E_bad=53.0875 d_man_ok=0.0009 d_man_bad=0.0013 stab_ok=0.9740 stab_bad=0.9740 structure=0/3
- E16_field_dynamics seed=921860 verdict=PASS: Dphi predicts transitions; rollout partial; RQM control separate notes=table_ok=true nn_next=B rqm_direct=1 rqm_compose=3 cdt_pred=1 cos_B=1.000 rollB=1.000 rollC=1.000 rollD=1.000 static_cosAB=0.203 causal_pre=0.186 audit_clean_for_AtoD_train=true
- E17_never_observed_states seed=921860 verdict=PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) notes=dist_C=0.0012 dist_B=0.0199 comp_dist=-0.0000 rqm_direct_edge=false rqm_compose_pred=2 audit_train=false audit_rqm_compose=true audit_bank=false valid_unseen=true
- E11_trainable_field_encoder seed=921861 verdict=PASS: trained encoder geometry > random projector notes=raw_margin=0.000 rand_margin=0.019 pre_margin=-0.001 post_margin=0.866 raw_knn=0.308 rand_knn=0.231 knn=1.000 para=1.000 sil=0.926 stab=0.990 causal_delta=1.038 loss_end=4.1095 elapsed_ms=42 raw_intra=1.000 raw_inter=1.000
- E12_crosslingual_concept seed=921861 verdict=SKIPPED_NO_GGUF notes=GEMMA2_GGUF unset y no hay .gguf bajo models/; harness exige FrozenGemma2Probe real. NO se afirma cross-lingual con léxico. Tests compilados; ejecución omitida honestamente.
- E13_relational_field seed=921861 verdict=PARTIAL: dynamic/static field generaliza holdout (RQM solo control) notes=A_table seen=1.00 un=0.00; B_RQM seen=0.60 un=0.00; C_static seen=0.20 un=0.33; D_dynamic seen=0.60 un=0.33
- E14_incremental_forgetting seed=921861 verdict=PASS: incremental recall retained; adversarial retain>=0.5 notes=dR_A=-0.003 dR_B=-0.002 dR_C=-0.000 final_acc=1.000 adv_retain_ABC=1.000 curve_A=Some([0.9993761919807417, 0.9989519890055237, 0.9990548057698526, 0.9960443397448651]) curve_B=Some([1.0, 0.999591824306257, 0.9976681087197727]) local_bias=1
- E15_semantic_without_labels seed=921861 verdict=FAIL: no structural signal for unseen incompatible combos notes=E_ok=53.9775 E_bad=53.7768 d_man_ok=0.0013 d_man_bad=0.0012 stab_ok=0.9770 stab_bad=0.9770 structure=0/3
- E16_field_dynamics seed=921861 verdict=WEAK: mild dynamics; may not beat static/NN notes=table_ok=true nn_next=B rqm_direct=1 rqm_compose=3 cdt_pred=1 cos_B=0.411 rollB=0.411 rollC=0.599 rollD=0.995 static_cosAB=0.995 causal_pre=0.991 audit_clean_for_AtoD_train=true
- E17_never_observed_states seed=921861 verdict=PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) notes=dist_C=0.0010 dist_B=0.0126 comp_dist=0.0000 rqm_direct_edge=false rqm_compose_pred=2 audit_train=false audit_rqm_compose=true audit_bank=false valid_unseen=true
- E11_trainable_field_encoder seed=921862 verdict=PASS: trained encoder geometry > random projector notes=raw_margin=-0.000 rand_margin=0.018 pre_margin=0.015 post_margin=0.936 raw_knn=0.385 rand_knn=0.462 knn=1.000 para=0.833 sil=0.942 stab=0.991 causal_delta=0.807 loss_end=0.8200 elapsed_ms=31 raw_intra=1.000 raw_inter=1.000
- E12_crosslingual_concept seed=921862 verdict=SKIPPED_NO_GGUF notes=GEMMA2_GGUF unset y no hay .gguf bajo models/; harness exige FrozenGemma2Probe real. NO se afirma cross-lingual con léxico. Tests compilados; ejecución omitida honestamente.
- E13_relational_field seed=921862 verdict=PARTIAL: dynamic/static field generaliza holdout (RQM solo control) notes=A_table seen=1.00 un=0.00; B_RQM seen=0.60 un=0.00; C_static seen=0.20 un=0.33; D_dynamic seen=0.20 un=0.33
- E14_incremental_forgetting seed=921862 verdict=PASS: incremental recall retained; adversarial retain>=0.5 notes=dR_A=0.013 dR_B=-0.000 dR_C=-0.002 final_acc=1.000 adv_retain_ABC=1.000 curve_A=Some([0.9812682310211687, 0.9992990463121467, 0.9991375716770057, 0.9943050251759076]) curve_B=Some([1.0, 0.9997777661299403, 0.9999995908169221]) local_bias=1
- E15_semantic_without_labels seed=921862 verdict=PARTIAL: weak structural separation without class labels notes=E_ok=54.9595 E_bad=55.0332 d_man_ok=0.0015 d_man_bad=0.0012 stab_ok=0.9777 stab_bad=0.9777 structure=1/3
- E16_field_dynamics seed=921862 verdict=PASS: Dphi predicts transitions; rollout partial; RQM control separate notes=table_ok=true nn_next=B rqm_direct=1 rqm_compose=3 cdt_pred=1 cos_B=1.000 rollB=1.000 rollC=1.000 rollD=1.000 static_cosAB=0.354 causal_pre=0.384 audit_clean_for_AtoD_train=true
- E17_never_observed_states seed=921862 verdict=PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) notes=dist_C=0.0017 dist_B=0.0112 comp_dist=0.0000 rqm_direct_edge=false rqm_compose_pred=2 audit_train=false audit_rqm_compose=true audit_bank=false valid_unseen=true
- E11_trainable_field_encoder seed=921863 verdict=PARTIAL: geometry improved but not all criteria notes=raw_margin=0.000 rand_margin=-0.005 pre_margin=-0.002 post_margin=0.857 raw_knn=0.231 rand_knn=0.231 knn=0.692 para=0.833 sil=0.888 stab=0.991 causal_delta=0.951 loss_end=2.2497 elapsed_ms=31 raw_intra=1.000 raw_inter=1.000
- E12_crosslingual_concept seed=921863 verdict=SKIPPED_NO_GGUF notes=GEMMA2_GGUF unset y no hay .gguf bajo models/; harness exige FrozenGemma2Probe real. NO se afirma cross-lingual con léxico. Tests compilados; ejecución omitida honestamente.
- E13_relational_field seed=921863 verdict=PARTIAL: dynamic/static field generaliza holdout (RQM solo control) notes=A_table seen=1.00 un=0.00; B_RQM seen=0.60 un=0.00; C_static seen=0.20 un=0.33; D_dynamic seen=0.20 un=0.33
- E14_incremental_forgetting seed=921863 verdict=PASS: incremental recall retained; adversarial retain>=0.5 notes=dR_A=0.033 dR_B=-0.000 dR_C=-0.002 final_acc=1.000 adv_retain_ABC=1.000 curve_A=Some([0.9616787530018774, 0.9993672421762679, 0.9987830312412241, 0.9946039153928422]) curve_B=Some([0.9999999999999988, 0.9994413270818527, 0.9999995376273079]) local_bias=1
- E15_semantic_without_labels seed=921863 verdict=PARTIAL: weak structural separation without class labels notes=E_ok=63.5001 E_bad=63.6111 d_man_ok=0.0020 d_man_bad=0.0015 stab_ok=0.9819 stab_bad=0.9819 structure=1/3
- E16_field_dynamics seed=921863 verdict=PASS: Dphi predicts transitions; rollout partial; RQM control separate notes=table_ok=true nn_next=B rqm_direct=1 rqm_compose=3 cdt_pred=1 cos_B=1.000 rollB=1.000 rollC=1.000 rollD=1.000 static_cosAB=0.150 causal_pre=0.163 audit_clean_for_AtoD_train=true
- E17_never_observed_states seed=921863 verdict=PASS: Dphi two-step + ToR approach never-stored A->C (RQM compose flagged separately) notes=dist_C=0.0020 dist_B=0.0092 comp_dist=-0.0000 rqm_direct_edge=false rqm_compose_pred=2 audit_train=false audit_rqm_compose=true audit_bank=false valid_unseen=true


## Controles y honestidad
- E8–E10 permanecen como baseline en `docs/resultados_experimentos_8_9_10.*` (no mezclados).
- E12: si no hay GGUF → `SKIPPED_NO_GGUF` (no se afirma cross-lingual con léxico).
- RQM es control; no explicación primaria de E13/E16/E17.
- Auditoría anti-leakage en E16/E17.
- No se presenta como evidencia de cognición / AGI.
- LLM congelado; solo se entrenan θ del encoder y φ de la dinámica.

## Resumen de veredictos (8 semillas)

- **E11_trainable_field_encoder**: PASS×6, PARTIAL×2
- **E12_crosslingual_concept**: SKIPPED_NO_GGUF×8
- **E13_relational_field**: PARTIAL×8
- **E14_incremental_forgetting**: PASS×8
- **E15_semantic_without_labels**: PARTIAL×5, FAIL×3
- **E16_field_dynamics**: PASS×6, WEAK×2
- **E17_never_observed_states**: PASS×8

## Negativos / límites (honestos)

- **E12**: `SKIPPED_NO_GGUF` — `GEMMA2_GGUF` ausente; no se afirma invariancia cross-lingual con léxico.
- **E13**: holdout (lobo→animal) débil con periferia lexicon; RQM control no explica el campo. Señal seen parcial vía top-2 multi-etiqueta.
- **E15**: separación estructural inconsistente entre semillas; métrica de energía/distancia, no accuracy binaria.
- **E16**: 2/8 semillas WEAK (dinámica no supera baseline en esa semilla).
- No se interpreta como cognición / AGI / consciencia.

## Mejoras de arquitectura aplicadas en esta corrida

1. `TrainableFieldEncoder` con prototipos fijos por familia + micro-rehearsal (E11).
2. Sesgos locales por concepto + replay (E14) frente a olvido catastrófico.
3. Dinámica `FieldDynamics` Dφ + auditoría anti-leakage (E16/E17).
4. Modos `STATIC_FIELD` / `DYNAMIC_FIELD` / `FIELD_PLUS_CDT`; RQM solo como control.
5. E12 harness exige `FrozenGemma2Probe` / GGUF real.

## Confirmación git

- Resultados en `docs/resultados_experimentos_11_17.md` y `.csv` (no mezclar con E8–E10).
- Push solo a `origin/exp/liquid-inference-experiments-8-9-10`.
- **main no modificado.**
