# Resultados — Etapa 2 Clean-Room v3 LONG

Protocolo: `docs/plan_autonomia_campo_v3.md` + `docs/etapa_2_autonomia_campo_experimentos.md` §29+.
Módulo: `src/field_autonomy_stage2_v2.rs`.
Rama: `exp/field-autonomy-next`.
Commit tip: `TIP_PLACEHOLDER`.
Seeds: 16 (base DEV 0xA300..0xA307; extra=true)
Seeds confirmación `0xB300–0xB30F`: **no corridas**.
Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.

## Hardening en esta corrida (v3.5)

- `HyperparamLock::long()`: dyn_epochs=560 dyn_updates=7 (push absolute cos_dyn).
- **E21 dual-probe (v3.5)**: SoftScale train/dyn + Relative-only static probe.
- **E22 (v3.5)**: compose-chain hop-2 Dφ; adaptive lin∩mlp; Relative static endpoints; stronger OOD decoder.
- **E23 (v3.5)**: 70/30 short/long TF; light h16 residual free-run (no h32/h64 free-run).
- Dual FeatPath retained: SoftScale E22/E23 dyn; Relative E18/E24/E25/E27 (+ E21/E22 static probe).
- Static baseline **sin** action + contraste geom-only; dyn extras + multi-familia + E21 dx denso.

Wall time: **1163.8s**. Filas: 240.

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
| `E18C_cdt_conditions_C0_C6` | NULL:2, PARTIAL:13, POSITIVE:1 |
| `E18_reinforced_rule_learning` | NEGATIVE:2, NULL:3, PARTIAL:6, POSITIVE:4, STRONG_POSITIVE:1 |
| `E19_provenance_audit` | NEGATIVE:3, NULL:4, PARTIAL:5, POSITIVE:2, STRONG_POSITIVE:2 |
| `E20_rule_vs_trajectory_antimem` | PARTIAL:16 |
| `E21_interpolation_vs_extrapolation` | NULL:1, PARTIAL:5, POSITIVE:1, STRONG_POSITIVE:9 |
| `E22_composition_rqm_off` | NEGATIVE:2, NULL:5, PARTIAL:3, POSITIVE:5, STRONG_POSITIVE:1 |
| `E23_rollout_no_teacher_forcing` | PARTIAL:12, POSITIVE:2, STRONG_POSITIVE:2 |
| `E24_static_vs_dynamic_paired` | NEGATIVE:4, NULL:2, PARTIAL:4, POSITIVE:6 |
| `E25_central_experience_cdt_cycle` | NULL:3, PARTIAL:9, POSITIVE:4 |
| `E26_experience_ablation_A_G` | PARTIAL:16 |
| `E27_rule_transfer` | NEGATIVE:2, NULL:1, PARTIAL:8, POSITIVE:5 |
| `E28_continual_forgetting` | PARTIAL:16 |
| `E29_causal_intervention` | NULL:1, PARTIAL:15 |
| `E30_serialize_reload_persistence` | PARTIAL:16 |
| `FASE_A_cleanroom_field_only` | POSITIVE:16 |

CSV: [`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)

## Lectura honesta (corrida LONG v3.5)

- Leakage FIELD_ONLY = **0**; contaminación DATASET_INVALID = **0**.
- `FASE_A` POSITIVE:16 (scaffold).
- Palancas v3.5: **E21 SoftScale-train + Relative-static probe**; **E22 compose-chain hop-2 + adaptive lin∩mlp + Relative static**; **E23 70/30 + light h16 residual free-run**.
- Seeds confirmación `0xB300–0xB30F`: **no corridas**.

### Comparación vs LONG v3.4 (tip `316cf7d` / content `09b83c1`)

| Exp | Antes (v3.4) | Ahora (v3.5) |
|---|---|---|
| E18 | NULL:4 PARTIAL:7 POSITIVE:4 STRONG:1 | NEGATIVE:2 NULL:3 PARTIAL:6 POSITIVE:4 STRONG:1 (ok, leve merma) |
| E21 | NEGATIVE:2 NULL:9 PARTIAL:5 (sin POSITIVE) | NULL:1 PARTIAL:5 POSITIVE:1 **STRONG:9** |
| E22 | NEGATIVE:15 NULL:1 (e2e~0.72; static SoftScale alto) | NEGATIVE:2 NULL:5 PARTIAL:3 **POSITIVE:5 STRONG:1** |
| E23 | PARTIAL:13 POSITIVE:1 STRONG:2 | PARTIAL:12 POSITIVE:2 STRONG:2 (similar) |
| E24 | NEGATIVE:2 NULL:4 PARTIAL:4 POSITIVE:6 | NEGATIVE:4 NULL:2 PARTIAL:4 POSITIVE:6 |
| E25 | NULL:2 PARTIAL:11 POSITIVE:3 | NULL:3 PARTIAL:9 **POSITIVE:4** |
| E27 | NEGATIVE:2 NULL:1 PARTIAL:8 POSITIVE:5 | NEGATIVE:2 NULL:1 PARTIAL:8 POSITIVE:5 |
| E28 | PARTIAL:16 | PARTIAL:16 (ok) |

### Números reales (cuellos / ganancias)

- **E21 recovered**: SoftScale dyn mean **0.881**; Relative-static mean **0.539**; Δmean dyn−static **+0.342** → **STRONG:9 / POSITIVE:1**. Cierra la regresión Relative-train de v3.4.
- **E22 verdict flip**: e2e mean **0.674** (v3.4 0.721; leve merma absoluta); Relative-static mean **0.305**; **14/16** e2e>static; pt_err mean **6.07** (antes 5.38); step1 **0.948**; oracle-mid **0.742**. Verdict hist pasa de casi-todo NEGATIVE a **POSITIVE/STRONG:6 + PARTIAL:3**.
- **E23 horizons**: mean cos h1..h64 = **[0.979, 0.950, 0.853, 0.684, 0.469, 0.325, 0.448]**; seeds ≥0.7: **[16,16,12,11,9,7,9]**. Short mostly intact; h32/h64 still <0.7 mean (h64 0.36→0.45; h32 0.40→0.32 — nudge mixto).
- **E18/E24/E25/E27**: POSITIVE cluster retained (E18 POSITIVE+STRONG:5; E24 POSITIVE:6; E25 POSITIVE:4; E27 POSITIVE:5).
- **E28**: PARTIAL:16 mantenido.

### Gaps que quedan

1. E22 aún tiene NEGATIVE:2 / NULL:5 — e2e absoluto bajó vs v3.4; need more OOD mid fidelity without SoftScale-static return.
2. E23 h32/h64 mean still <0.7; 70/30+h16 residual no cerró el horizonte largo (h8 también −0.04).
3. E18/E24 leve merma vs v3.4 (más NEGATIVE en E18/E24) — monitor dual-path side effects.
4. Confirmation `0xB300–0xB30F` still deferred (E21/E22 now look confirmation-ready; E23 long still weak).

### Próximas palancas

1. E22: residual skip on abs SoftScale channels in mid decode; optional geom-compose loss without raising Relative-static.
2. E23: true residual state skip inside Dφ step (α·Wψ+(1−α)·ψ) annealed by horizon; or scheduled free-run after h8 only when h1–h8 DEV≥0.9.
3. Confirmation seeds once E23 long ≥0.55 mean or accepted as residual gap.
