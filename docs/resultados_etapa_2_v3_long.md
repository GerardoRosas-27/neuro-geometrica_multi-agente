# Resultados — Etapa 2 Clean-Room v3 LONG

Protocolo: `docs/plan_autonomia_campo_v3.md` + `docs/etapa_2_autonomia_campo_experimentos.md` §29+.
Módulo: `src/field_autonomy_stage2_v2.rs`.
Rama: `exp/field-autonomy-next`.
Commit tip: `09b83c1cb233a25de2b89ba9daaf703d80d706f2`.
Seeds: 16 (base DEV 0xA300..0xA307; extra=true)
Seeds confirmación `0xB300–0xB30F`: **no corridas**.
Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.

## Hardening en esta corrida (v3.4)

- `HyperparamLock::long()`: dyn_epochs=560 dyn_updates=7 (push absolute cos_dyn).
- **Dual FeatPath (v3.4)**: SoftScale for E22 mid/compose + E23 rollout; Relative/centered for E18/E21/E24/E25/E27 static desaturation.
- Static baseline **sin** action + contraste geom-only en encoder.
- Dyn-focused extras + low-cos curriculum repeats.
- E23: mixed h1–h8 TF+free-run + richer long TF (h16/h32/h64, ~22%); no free-run on long.
- E22: SoftScale + OOD decoder (rule-roll far domain + lin∩mlp blend); Dφ frozen; e2e primary.
- Currícula multi-familia + E21 dx denso.

Wall time: **992.0s**. Filas: 240.

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
| `E18C_cdt_conditions_C0_C6` | NULL:1, PARTIAL:15 |
| `E18_reinforced_rule_learning` | NULL:4, PARTIAL:7, POSITIVE:4, STRONG_POSITIVE:1 |
| `E19_provenance_audit` | NEGATIVE:3, NULL:4, PARTIAL:7, POSITIVE:2 |
| `E20_rule_vs_trajectory_antimem` | PARTIAL:16 |
| `E21_interpolation_vs_extrapolation` | NEGATIVE:2, NULL:9, PARTIAL:5 |
| `E22_composition_rqm_off` | NEGATIVE:15, NULL:1 |
| `E23_rollout_no_teacher_forcing` | PARTIAL:13, POSITIVE:1, STRONG_POSITIVE:2 |
| `E24_static_vs_dynamic_paired` | NEGATIVE:2, NULL:4, PARTIAL:4, POSITIVE:6 |
| `E25_central_experience_cdt_cycle` | NULL:2, PARTIAL:11, POSITIVE:3 |
| `E26_experience_ablation_A_G` | NULL:1, PARTIAL:15 |
| `E27_rule_transfer` | NEGATIVE:2, NULL:1, PARTIAL:8, POSITIVE:5 |
| `E28_continual_forgetting` | PARTIAL:16 |
| `E29_causal_intervention` | NULL:2, PARTIAL:14 |
| `E30_serialize_reload_persistence` | PARTIAL:16 |
| `FASE_A_cleanroom_field_only` | POSITIVE:16 |

CSV: [`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)

## Lectura honesta (corrida LONG v3.4)

- Leakage FIELD_ONLY = **0**; contaminación DATASET_INVALID = **0**.
- `FASE_A` POSITIVE:16 (scaffold).
- Palancas v3.4: **dual FeatPath** (SoftScale≠Relative); E22 OOD mid decoder (rule-roll + lin∩mlp); E23 long-TF nudge (h16/h32/h64); Relative static desat on E18/E21/E24/E25/E27.
- Seeds confirmación `0xB300–0xB30F`: **no corridas**.

### Comparación vs LONG v3.3 (tip `8d3d71e` / content `b612524`)

| Exp | Antes (v3.3) | Ahora (v3.4) |
|---|---|---|
| E18 | NEGATIVE:1 NULL:7 PARTIAL:8 (sin POSITIVE; static~0.97) | NULL:4 PARTIAL:7 **POSITIVE:4 STRONG:1** (static desat) |
| E21 | NEGATIVE:3 NULL:1 PARTIAL:6 POSITIVE:3 STRONG:3 | NEGATIVE:2 NULL:9 PARTIAL:5 (regresión absolutos) |
| E22 | NEGATIVE:15 NULL:1 (oracle-mid~0.76; e2e~0.65; pt_err~7.1) | NEGATIVE:15 NULL:1 (oracle-mid~0.73; **e2e~0.72**; **pt_err~5.4**) |
| E23 | PARTIAL:14 POSITIVE:2 | PARTIAL:13 POSITIVE:1 **STRONG:2** (h32/h64↑) |
| E24 | NEGATIVE:4 NULL:11 POSITIVE:1 | NEGATIVE:2 NULL:4 PARTIAL:4 **POSITIVE:6** |
| E25 | NULL:13 PARTIAL:3 | NULL:2 PARTIAL:11 **POSITIVE:3** |
| E27 | NEGATIVE:3 NULL:13 | NEGATIVE:2 NULL:1 PARTIAL:8 **POSITIVE:5** |
| E28 | PARTIAL:16 | PARTIAL:16 (ok) |

### Números reales (cuellos / ganancias)

- **Static desaturation restored (Relative path)**: E18 static mean **0.683** (v3.3 soft-scale era **0.973**); Δmean dyn−static **+0.127**, **8/16** Δ>+0.05 → POSITIVE/STRONG vuelven.
- **E24/E25/E27**: margen dyn−static recuperado (E24 POSITIVE:6; E25 POSITIVE:3; E27 POSITIVE:5) — el tradeoff single-vector de v3.3 queda cerrado vía dual path.
- **E22 e2e OOD mejor**: e2e mean **0.721** (antes 0.651); pt_err mean **5.38** (antes 7.10); step1 mean **0.966**. Oracle-mid mean **0.731** (antes 0.759; leve merma aceptable). Verdict hist aún NEGATIVE (static compose endpoints siguen altos).
- **E23 horizons**: mean cos h1..h64 = **[0.986, 0.952, 0.867, 0.728, 0.526, 0.402, 0.364]**; seeds ≥0.7: **[16,16,15,11,7,8,7]** (v3.3 era [16,16,14,12,8,7,4]). Short horizons intactos; **h32 0.05→0.40**, **h64 0.05→0.36**.
- **E21 regresión**: Relative en E21 baja absolutos (dyn mean 0.588; sin POSITIVE). Gap: E21 quizás necesite SoftScale para dyn o dual-encoder (Relative solo en static score).
- **E28**: PARTIAL:16 mantenido.

### Gaps que quedan

1. E22 verdict aún NEGATIVE (e2e < static on compose geom); need larger OOD gap or static compose desat without killing SoftScale mid.
2. E21 lost STRONG/POSITIVE vs v3.3 — consider SoftScale train + Relative-only static probe for E21.
3. E23 h32/h64 improved but still <0.7 mean; residual/skip in Dφ still open.
4. Confirmation `0xB300–0xB30F` still deferred (E18/E24 POSITIVE now visible but E21 unstable).

### Próximas palancas

1. E21 dual-probe: SoftScale train/dyn + Relative-only static baseline (per-metric path).
2. E22: stronger OOD calibration / soft-scale residual skip on abs channels; optional geom-static desat for compose endpoints.
3. E23: residual state skip or scheduled free-run after h8 stable.
4. Confirmation seeds only after E21 recovered.
