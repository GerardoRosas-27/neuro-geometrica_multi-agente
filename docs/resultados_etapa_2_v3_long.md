# Resultados — Etapa 2 Clean-Room v3 LONG

Protocolo: `docs/plan_autonomia_campo_v3.md` + `docs/etapa_2_autonomia_campo_experimentos.md` §29+.
Módulo: `src/field_autonomy_stage2_v2.rs`.
Rama: `exp/field-autonomy-next`.
Commit tip: `01db3af7782cfc870eafe60192412ef15cb3116a`.
Seeds: 16 (base DEV 0xA300..0xA307; extra=true)
Seeds confirmación `0xB300–0xB30F`: **no corridas**.
Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.

## Hardening en esta corrida

- `HyperparamLock::long()`: más datos/épocas/updates.
- Static baseline **sin** action channels (geom-only) para margen dyn−static.
- Real multi-step unroll train (teacher-forced + free-run h=2/4/8).
- E22: decoder mid end-to-end compose (sin oracle-mid); oracle en notes.
- Currícula multi-familia (trans/rot/scale/affine) mezclada en TRAIN.
- E21: curriculum dx denso + multifamily mix.

Wall time: **586.5s**. Filas: 240.

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
| `E18C_cdt_conditions_C0_C6` | NULL:13, PARTIAL:1, POSITIVE:2 |
| `E18_reinforced_rule_learning` | NULL:11, PARTIAL:5 |
| `E19_provenance_audit` | NULL:12, PARTIAL:4 |
| `E20_rule_vs_trajectory_antimem` | PARTIAL:16 |
| `E21_interpolation_vs_extrapolation` | NULL:11, PARTIAL:5 |
| `E22_composition_rqm_off` | NEGATIVE:4, NULL:12 |
| `E23_rollout_no_teacher_forcing` | PARTIAL:14, POSITIVE:2 |
| `E24_static_vs_dynamic_paired` | NEGATIVE:2, NULL:14 |
| `E25_central_experience_cdt_cycle` | NULL:16 |
| `E26_experience_ablation_A_G` | NULL:12, PARTIAL:1, POSITIVE:3 |
| `E27_rule_transfer` | NEGATIVE:7, NULL:9 |
| `E28_continual_forgetting` | PARTIAL:16 |
| `E29_causal_intervention` | PARTIAL:16 |
| `E30_serialize_reload_persistence` | PARTIAL:16 |
| `FASE_A_cleanroom_field_only` | POSITIVE:16 |

CSV: [`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)

## Lectura honesta (corrida LONG)

- Leakage FIELD_ONLY = **0**; contaminación DATASET_INVALID = **0**.
- `FASE_A` POSITIVE:16 (scaffold).
- Palancas v3.1 aplicadas: static geom-only (sin action); unroll real h=2/4/8; decoder compose e2e; multifamily mix en TRAIN; soft-scale features (sin L2 total).

### Comparación vs LONG previa (tip `14d5f58` / docs stamp `1fc612b`)

| Exp | Antes | Ahora |
|---|---|---|
| E18 | NEGATIVE:3 NULL:13 | **NULL:11 PARTIAL:5** (0 NEGATIVE) |
| E21 | NEGATIVE:3 NULL:10 PARTIAL:3 | **NULL:11 PARTIAL:5** (0 NEGATIVE) |
| E22 | NULL:15 PARTIAL:1 (oracle-mid) | NEGATIVE:4 NULL:12 (**métrica e2e-decoder**, más dura) |
| E23 | PARTIAL:15 POSITIVE:1 | **PARTIAL:14 POSITIVE:2** |
| E24 | NULL:16 | NEGATIVE:2 NULL:14 |

### Cuellos restantes (números reales)

- **Static aún saturado** (~0.99 geom-only en TEST lejano): mean Δ(dyn−static) E18≈−0.004, E21≈−0.002, E24≈−0.012; **0/16** cruzan margen +0.05 POSITIVE.
- **E22**: oracle-mid sigue alto (~0.94–0.98 en notes) pero **decoder mid** falla (pt_err≈4–7); e2e-decoder es el cuello, no el chaining de Dφ.
- **E23**: h1–h4 fuertes (mean cos 0.99/0.97/0.88); colapso residual h32–h64 (mean 0.23/0.18) aunque 5/16 seeds mantienen cos≥0.7 a h64.
- E25 NULL:16; E27 NEGATIVE:7 — transferencia/experiencia aún frágiles.

### Próximas palancas

1. Decoder no lineal / joint train decoder↔Dφ (bajar pt_err compose).
2. Features relativas / centradas (romper saturación static en región TEST |x|≫|Δ|).
3. Unroll loss con BPTT corto o curriculum horizon annealing hacia h32.
4. Confirmation `0xB300–0xB30F` solo tras DEV con POSITIVE estable en E18/E21/E24.
