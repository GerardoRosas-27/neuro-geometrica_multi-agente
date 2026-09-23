# Resultados — Etapa 2 Clean-Room v2

Protocolo: `docs/etapa_2_autonomia_campo_experimentos.md` **§29+** (prevalece).
Módulo: `src/field_autonomy_stage2_v2.rs` (legacy `field_autonomy_stage2.rs` = histórico / pre-cleanroom).
Rama: `exp/field-autonomy-stage-2`
Seeds desarrollo: 8 × `0xA300..0xA307` (full suite)
Seeds confirmación `0xB300–0xB30F`: **reservadas / diferidas** (no corridas; lock solo sobre DEV).
Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.

## Nota sobre resultados previos

`docs/resultados_etapa_2_autonomia.md` / `.csv` son **pre-cleanroom / históricos** (seeds `0xE1800..`). No mezclar con este informe.

Wall time: **12.1s**. Filas: 120.

## Hyperparam lock

- Flujo: TRAIN→DEV→LOCK→TEST
- Status tras suite: **LOCKED**
- enc_epochs=60 dyn_epochs=180 dyn_updates=3 train_n/dev_n/test_n=28/10/12
- Confirmation no usada → TEST no invalidado por retune post-TEST.

## Contaminación

- Filas con `DATASET_INVALID`: **0**
- Auditor hard: exact/near/equivalent_state/equivalent_target; same_orbit & equivalent_transformation cubiertos en tests unitarios.

## Leakage

- Filas con leakage>0: **0** (objetivo FIELD_ONLY = 0)

## Histogramas de veredicto por experimento

| Experimento | verdict hist |
|---|---|
| `E18C_cdt_conditions_C0_C6` | NULL:8 |
| `E18_reinforced_rule_learning` | NULL:2, PARTIAL:6 |
| `E19_provenance_audit` | NULL:1, PARTIAL:7 |
| `E20_rule_vs_trajectory_antimem` | PARTIAL:8 |
| `E21_interpolation_vs_extrapolation` | NEGATIVE:6, NULL:2 |
| `E22_composition_rqm_off` | NEGATIVE:2, NULL:2, PARTIAL:4 |
| `E23_rollout_no_teacher_forcing` | PARTIAL:8 |
| `E24_static_vs_dynamic_paired` | NULL:8 |
| `E25_central_experience_cdt_cycle` | NULL:8 |
| `E26_experience_ablation_A_G` | NULL:8 |
| `E27_rule_transfer` | NULL:7, PARTIAL:1 |
| `E28_continual_forgetting` | PARTIAL:8 |
| `E29_causal_intervention` | PARTIAL:8 |
| `E30_serialize_reload_persistence` | PARTIAL:8 |
| `FASE_A_cleanroom_field_only` | POSITIVE:8 |

## Implementado vs diferido

| Estado | Ítems |
|---|---|
| Landed | Generator+auditor+sealed manifests; E18; E18C C0–C6; E19 provenance; E20 antimem; E21 interp/extrap; E22 compose RQM-OFF; E23 rollout h≤64 no TF; E24 paired+bootstrap; E25 cycle; E26 ablation; E27 transfer; E28 continual; E29 intervene; E30 serialize/reload (in-process) |
| Deferred | Confirmation seeds 0xB300–0xB30F; true OS process restart in E30; Gemma linguistic periphery benchmark |

## Honestidad

- Leakage FIELD_ONLY = **0** en todas las filas de esta corrida.
- No se reutilizó estado aprendido E11–E17; encoder/D_phi re-init aleatorio por corrida.
- TEST sellado (sha256) antes de entrenar; confirmation seeds no tocadas.

CSV: [`resultados_etapa_2_cleanroom_v2.csv`](resultados_etapa_2_cleanroom_v2.csv)
