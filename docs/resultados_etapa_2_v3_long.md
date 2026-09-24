# Resultados — Etapa 2 Clean-Room v3 LONG

Protocolo: `docs/plan_autonomia_campo_v3.md` + `docs/etapa_2_autonomia_campo_experimentos.md` §29+.
Módulo: `src/field_autonomy_stage2_v2.rs`.
Rama: `exp/field-autonomy-next`.
Commit tip: `e2650806db5aaa9baba59a2a216c70eaf7e8e0a3`.
Seeds: 16 (base DEV 0xA300..0xA307; extra=true)
Seeds confirmación `0xB300–0xB30F`: **no corridas**.
Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.

## Hardening en esta corrida

- `HyperparamLock::long()`: más datos/épocas/updates.
- Features relativas/periódicas (frac + multi-freq) — anti saturación static.
- Static baseline **sin** action + contraste geom-only en encoder.
- Dyn-focused extra updates cuando static_cos alta.
- Horizon annealing → h32 (TF + free-run) para E23.
- E22: residual MLP PointDecoder joint con Dφ (sin oracle-mid; sin train compose).
- Currícula multi-familia + E21 dx denso.

Wall time: **973.6s**. Filas: 240.

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
| `E18C_cdt_conditions_C0_C6` | NULL:1, PARTIAL:14, POSITIVE:1 |
| `E18_reinforced_rule_learning` | NEGATIVE:2, NULL:4, PARTIAL:8, POSITIVE:2 |
| `E19_provenance_audit` | NEGATIVE:3, NULL:6, PARTIAL:4, POSITIVE:2, STRONG_POSITIVE:1 |
| `E20_rule_vs_trajectory_antimem` | NEGATIVE:1, NULL:1, PARTIAL:12, POSITIVE:2 |
| `E21_interpolation_vs_extrapolation` | NEGATIVE:3, NULL:10, PARTIAL:2, POSITIVE:1 |
| `E22_composition_rqm_off` | NEGATIVE:15, NULL:1 |
| `E23_rollout_no_teacher_forcing` | NULL:2, PARTIAL:13, POSITIVE:1 |
| `E24_static_vs_dynamic_paired` | NEGATIVE:6, NULL:8, PARTIAL:1, POSITIVE:1 |
| `E25_central_experience_cdt_cycle` | NULL:2, PARTIAL:12, POSITIVE:2 |
| `E26_experience_ablation_A_G` | PARTIAL:14, POSITIVE:2 |
| `E27_rule_transfer` | NEGATIVE:2, NULL:1, PARTIAL:8, POSITIVE:4, STRONG_POSITIVE:1 |
| `E28_continual_forgetting` | NULL:15, PARTIAL:1 |
| `E29_causal_intervention` | NULL:5, PARTIAL:11 |
| `E30_serialize_reload_persistence` | PARTIAL:16 |
| `FASE_A_cleanroom_field_only` | POSITIVE:16 |

CSV: [`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)

## Lectura honesta (corrida LONG)

- Leakage FIELD_ONLY = **0**; contaminación DATASET_INVALID = **0**.
- `FASE_A` POSITIVE:16 (scaffold).
- Palancas v3.2 aplicadas: features relativas/periódicas; contraste geom-only; dyn-focused extras; horizon anneal→h32; residual MLP PointDecoder joint (sin train compose).
- Seeds confirmación `0xB300–0xB30F`: **no corridas**.

### Comparación vs LONG v3.1 (tip `44aeeb4` / content `01db3af`)

| Exp | Antes (v3.1) | Ahora (v3.2) |
|---|---|---|
| E18 | NULL:11 PARTIAL:5 | **NEGATIVE:2 NULL:4 PARTIAL:8 POSITIVE:2** |
| E21 | NULL:11 PARTIAL:5 | NEGATIVE:3 NULL:10 PARTIAL:2 **POSITIVE:1** |
| E22 | NEGATIVE:4 NULL:12 | **NEGATIVE:15 NULL:1** (regresión e2e) |
| E23 | PARTIAL:14 POSITIVE:2 | NULL:2 PARTIAL:13 POSITIVE:1 |
| E24 | NEGATIVE:2 NULL:14 | NEGATIVE:6 NULL:8 PARTIAL:1 **POSITIVE:1** |
| E25 | NULL:16 | NULL:2 PARTIAL:12 **POSITIVE:2** |
| E27 | NEGATIVE:7 NULL:9 | NEGATIVE:2 NULL:1 PARTIAL:8 **POSITIVE:4 STRONG:1** |

### Números reales (cuellos / ganancias)

- **Static desaturado** (objetivo principal de features relativas): E18 static mean **0.678** (0/16 ≥0.95; antes ~0.99). E21 static mean **0.372**. E24 static mean **0.780** (1/16 ≥0.95).
- **Margen dyn−static**: E18 Δmean=**+0.061**, **7/16** con Δ>+0.05 (antes 0/16); E21 Δmean=**+0.184**, **12/16** Δ>+0.05; E24 Δmean=**−0.022**, solo **2/16** Δ>+0.05. POSITIVE de veredicto sigue exigiendo también cos_dyn≥0.85 → E18 solo 2/16 POSITIVE pese al margen.
- **E22**: oracle-mid mean **0.430** (antes ~0.94–0.98); step1 mean 0.781; **pt_err mean 7.76** (min 5.3 max 15.6). Decoder+features relativas empeoraron compose e2e; cuello sigue siendo decode/cadena mid.
- **E23**: mean cos h1..h64 = **[0.833, 0.680, 0.438, 0.201, 0.091, 0.246, 0.221]**; seeds cos≥0.7: [13,10,5,2,2,2,4]. Annealing→h32 no recuperó h32–h64 y debilitó h1–h4 vs v3.1 (0.99/0.97/0.88).
- **E25/E27**: mejora clara (experiencia/transfer); E28 pasó de PARTIAL:16 a NULL:15 (posible daño colateral).

### Próximas palancas

1. E22: decoder en espacio de acción / mid-state con features de acción preservadas; no degradar oracle-mid (~0.94).
2. E23: anneal sin sacrificar h1–h8 (replay corto + largo); residual skip en Dφ o BPTT corto.
3. Subir cos_dyn absoluto (E18/E24) ahora que el margen static ya no satura — más dyn_epochs o loss en punto vía decoder.
4. Confirmation `0xB300–0xB30F` solo tras POSITIVE estable en E18/E21/E24 (aún no).
