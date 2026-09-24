# Resultados — Etapa 2 Clean-Room v3 LONG

Protocolo: `docs/plan_autonomia_campo_v3.md` + `docs/etapa_2_autonomia_campo_experimentos.md` §29+.
Módulo: `src/field_autonomy_stage2_v2.rs`.
Rama: `exp/field-autonomy-next`.
Commit tip: `14d5f582ac3575489874f8c87438f6e6b7b69dcc`.
Seeds: 16 (base DEV 0xA300..0xA307; extra=true)
Seeds confirmación `0xB300–0xB30F`: **no corridas**.
Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.

## Hardening en esta corrida

- `HyperparamLock::long()`: más datos/épocas/updates.
- Action cues para translation/rotation/scaling/affine/compose.
- E21: curriculum dx más denso + augment.
- E22: eval secuencial oracle-mid (T1 luego T2) RQM-OFF; single-shot en notes.

Wall time: **225.4s**. Filas: 240.

## Hyperparam lock

- Status: **LOCKED**
- enc_epochs=160 dyn_epochs=560 dyn_updates=5 train_n/dev_n/test_n=80/20/24

## Contaminación

- Filas `DATASET_INVALID`: **0**

## Leakage

- Filas leakage>0: **0**

## Histogramas de veredicto

| Experimento | verdict hist |
|---|---|
| `E18C_cdt_conditions_C0_C6` | NULL:5, PARTIAL:7, POSITIVE:4 |
| `E18_reinforced_rule_learning` | NEGATIVE:3, NULL:13 |
| `E19_provenance_audit` | NEGATIVE:2, NULL:12, PARTIAL:2 |
| `E20_rule_vs_trajectory_antimem` | PARTIAL:16 |
| `E21_interpolation_vs_extrapolation` | NEGATIVE:3, NULL:10, PARTIAL:3 |
| `E22_composition_rqm_off` | NULL:15, PARTIAL:1 |
| `E23_rollout_no_teacher_forcing` | PARTIAL:15, POSITIVE:1 |
| `E24_static_vs_dynamic_paired` | NULL:16 |
| `E25_central_experience_cdt_cycle` | NULL:12, PARTIAL:2, POSITIVE:2 |
| `E26_experience_ablation_A_G` | NULL:9, PARTIAL:5, POSITIVE:2 |
| `E27_rule_transfer` | NULL:14, PARTIAL:2 |
| `E28_continual_forgetting` | PARTIAL:16 |
| `E29_causal_intervention` | NULL:2, PARTIAL:14 |
| `E30_serialize_reload_persistence` | PARTIAL:16 |
| `FASE_A_cleanroom_field_only` | POSITIVE:16 |

CSV: [`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)


## Lectura honesta (corrida LONG final)

- Leakage FIELD_ONLY = **0**; contaminación DATASET_INVALID = **0**.
- `FASE_A` POSITIVE:16 (scaffold).
- Señales nuevas vs cleanroom v2 baseline: **POSITIVE** en algunos seeds de `E18C`, `E23`, `E25`, `E26`.
- Cuello dominante: **static cosine saturado (~0.98–0.99)** → margen dyn−static casi nunca alcanza +0.05 (umbral POSITIVE), aunque cos_dyn sea alto.
- E21/E22: mejora vs NEGATIVE sistemático del baseline, pero aún PARTIAL/NULL/NEGATIVE residual; extrapolación dx=3 y composición end-to-end siguen frágiles.
- E23: h1–h8 fuertes; colapso en h32–h64 en muchas semillas (PARTIAL con 1 POSITIVE).

## Próximas palancas

1. Baseline static **sin** action channels (o action abortada) para medir ganancia causal de Dφ.
2. Decoder/reproyección para composición end-to-end sin oracle-mid.
3. Unroll training multi-step real (teacher-forced h=2..8) orientado a E23.
4. Currícula multi-familia (rot/scale/affine) balanceada; no solo translation.
5. Confirmation seeds `0xB300–0xB30F` solo tras lock DEV estable (aún diferidas).
