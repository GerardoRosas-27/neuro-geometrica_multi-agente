# Resultados — Etapa 2 Clean-Room v3 LONG

Protocolo: `docs/plan_autonomia_campo_v3.md` + `docs/etapa_2_autonomia_campo_experimentos.md` §29+.
Módulo: `src/field_autonomy_stage2_v2.rs`.
Rama: `exp/field-autonomy-next`.
Commit tip: `b6125245105ef1d132d6230c10028be4ab720dec`.
Seeds: 16 (base DEV 0xA300..0xA307; extra=true)
Seeds confirmación `0xB300–0xB30F`: **no corridas**.
Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.

## Hardening en esta corrida (v3.3)

- `HyperparamLock::long()`: dyn_epochs=560 dyn_updates=7 (push absolute cos_dyn).
- Features **híbridas**: soft-scale absolute (v3.1) + light frac/periodic (no relative collapse).
- Static baseline **sin** action + contraste geom-only en encoder.
- Dyn-focused extras + low-cos curriculum repeats.
- E23: mixed h1–h8 (TF+free-run) + sparse h16/h32 TF-only (no anneal-only→h32).
- E22: MLP decoder action-aware con Dφ **frozen** (preserve oracle-mid); e2e primary.
- Currícula multi-familia + E21 dx denso.

Wall time: **795.9s**. Filas: 240.

## Hyperparam lock

- Status: **LOCKED**
- enc_epochs=160 dyn_epochs=560 dyn_updates=7 train_n/dev_n/test_n=80/20/24

## Contaminación

- Filas `DATASET_INVALID`: **0**

## Leakage

- Filas leakage>0: **0**

## Histogramas de veredicto

| Experimento | verdict hist |
|---|---|
| `E18C_cdt_conditions_C0_C6` | NULL:2, PARTIAL:11, POSITIVE:3 |
| `E18_reinforced_rule_learning` | NEGATIVE:1, NULL:7, PARTIAL:8 |
| `E19_provenance_audit` | NEGATIVE:3, NULL:1, PARTIAL:11, POSITIVE:1 |
| `E20_rule_vs_trajectory_antimem` | PARTIAL:16 |
| `E21_interpolation_vs_extrapolation` | NEGATIVE:3, NULL:1, PARTIAL:6, POSITIVE:3, STRONG_POSITIVE:3 |
| `E22_composition_rqm_off` | NEGATIVE:15, NULL:1 |
| `E23_rollout_no_teacher_forcing` | PARTIAL:14, POSITIVE:2 |
| `E24_static_vs_dynamic_paired` | NEGATIVE:4, NULL:11, POSITIVE:1 |
| `E25_central_experience_cdt_cycle` | NULL:13, PARTIAL:3 |
| `E26_experience_ablation_A_G` | NULL:2, PARTIAL:9, POSITIVE:5 |
| `E27_rule_transfer` | NEGATIVE:3, NULL:13 |
| `E28_continual_forgetting` | PARTIAL:16 |
| `E29_causal_intervention` | NULL:1, PARTIAL:15 |
| `E30_serialize_reload_persistence` | PARTIAL:16 |
| `FASE_A_cleanroom_field_only` | POSITIVE:16 |

CSV: [`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)

## Lectura honesta (corrida LONG v3.3)

- Leakage FIELD_ONLY = **0**; contaminación DATASET_INVALID = **0**.
- `FASE_A` POSITIVE:16 (scaffold).
- Palancas v3.3: hybrid soft-scale features; action-aware decoder (Dφ frozen); mixed horizons; dyn_updates=7.
- Seeds confirmación `0xB300–0xB30F`: **no corridas**.

### Comparación vs LONG v3.2 (tip `5febe95` / content `e265080`)

| Exp | Antes (v3.2) | Ahora (v3.3) |
|---|---|---|
| E18 | NEGATIVE:2 NULL:4 PARTIAL:8 POSITIVE:2 | NEGATIVE:1 NULL:7 PARTIAL:8 (sin POSITIVE; static re-satura) |
| E21 | NEGATIVE:3 NULL:10 PARTIAL:2 POSITIVE:1 | NEGATIVE:3 NULL:1 PARTIAL:6 **POSITIVE:3 STRONG:3** |
| E22 | NEGATIVE:15 NULL:1 (oracle-mid~0.43) | NEGATIVE:15 NULL:1 (oracle-mid **↑0.76**; e2e aún duro) |
| E23 | NULL:2 PARTIAL:13 POSITIVE:1 | PARTIAL:14 **POSITIVE:2** (short horizons recuperados) |
| E24 | NEGATIVE:6 NULL:8 PARTIAL:1 POSITIVE:1 | NEGATIVE:4 NULL:11 POSITIVE:1 |
| E25 | NULL:2 PARTIAL:12 POSITIVE:2 | NULL:13 PARTIAL:3 (regresión) |
| E27 | NEGATIVE:2 NULL:1 PARTIAL:8 POSITIVE:4 STRONG:1 | NEGATIVE:3 NULL:13 (regresión) |
| E28 | NULL:15 PARTIAL:1 | **PARTIAL:16** (restaurado) |

### Números reales (cuellos / ganancias)

- **E22 oracle-mid restaurado**: mean **0.759** (antes 0.430); **8/16 ≥0.85**. step1 mean 0.960. e2e mean 0.651; pt_err mean 7.10. Dφ frozen + soft-scale hybrid evita el colapso relative→oracle-mid; decode e2e sigue siendo el cuello.
- **E23 short horizons**: mean cos h1..h64 = **[0.990, 0.971, 0.860, 0.631, 0.313, 0.053, 0.046]**; seeds cos≥0.7: **[16,16,14,12,8,7,4]** (v3.2 era [13,10,5,2,2,2,4]; v3.1 h1–h4 ~0.99/0.97/0.88). Mixed short+sparse long + TF-only en h16/h32 recupera h1–h8; h32–h64 aún frágil.
- **Absolute cos_dyn**: E18 dyn mean **0.972** (16/16 ≥0.85); E21 dyn mean **0.925** (15/16 ≥0.85); E24 dyn mean **0.919** (14/16 ≥0.85). Umbral absoluto OK; el margen dyn−static es el limitante.
- **Static re-saturado** (tradeoff vs v3.2 relative): E18 static mean **0.973** (14/16 ≥0.95; v3.2 era 0.678). Δmean E18 **−0.001**, 0/16 Δ>+0.05. Soft-scale preserva compose pero pierde la desaturación que daba POSITIVE en E18/E25/E27.
- **E21**: ganancia clara (POSITIVE+STRONG) con dyn margin Δmean=+0.065, 6/16 Δ>+0.05.
- **E28**: restaurado a PARTIAL:16 (v3.2 NULL:15).

### Próximas palancas

1. Dual-feature or dual-encoder: soft-scale path for E22 mid decode + relative path for static baseline desaturation (avoid single-vector tradeoff).
2. E22 e2e: better OOD mid decoder (train-region [-2,2] → TEST [3,6]); residual skip tied to soft-scale abs channels.
3. E23: residual/skip in Dφ or scheduled free-run only after short TF is stable; h32–h64 still weak.
4. Confirmation `0xB300–0xB30F` still deferred (E18/E24 POSITIVE not stable).
