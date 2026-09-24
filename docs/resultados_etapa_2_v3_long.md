# Resultados — Etapa 2 Clean-Room v3 LONG

Protocolo: `docs/plan_autonomia_campo_v3.md` + `docs/etapa_2_autonomia_campo_experimentos.md` §29+.
Módulo: `src/field_autonomy_stage2_v2.rs`.
Rama: `exp/field-autonomy-next`.
Commit tip: `daa18dcf27cb77ee9f4300231b0206ae49f24f1a`.
Seeds: 16 (base DEV 0xA300..0xA307; extra=true)
Seeds confirmación `0xB300–0xB30F`: **no corridas**.
Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.

## Hardening en esta corrida (v3.6)

- `HyperparamLock::long()`: dyn_epochs=560 dyn_updates=7 (push absolute cos_dyn).
- **E21 dual-probe retained**: SoftScale train/dyn + Relative-only static probe.
- **E22 (v3.6)**: disagree-aware lin∩mlp SoftScale-abs; encode-consistency mid refine; closed-loop hop-2.
- **E23 (v3.6)**: shared v3.5 mix retained (no local long TF; avoid h64 regression); pure step eval.
- Dual FeatPath retained: SoftScale E22/E23 dyn; Relative E18/E24/E25/E27 (+ E21/E22 static probe).
- Static baseline **sin** action + contraste geom-only; dyn extras + multi-familia + E21 dx denso.

Wall time: **1128.0s**. Filas: 240.

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
| `E22_composition_rqm_off` | NEGATIVE:1, NULL:3, PARTIAL:3, POSITIVE:8, STRONG_POSITIVE:1 |
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

## Lectura honesta (corrida LONG v3.6)

- Leakage FIELD_ONLY = **0**; contaminación DATASET_INVALID = **0**.
- `FASE_A` POSITIVE:16 (scaffold).
- Palancas v3.6: **E22 encode-consistency mid refine + disagree-aware lin∩mlp SoftScale-abs + closed-loop hop-2 + denser compose-chain**; **E23 shared v3.5 mix retained** (local long TF tried in diag, hurt h64 — reverted); **E21 SoftScale+Relative probe retained**.
- Seeds confirmación `0xB300–0xB30F`: **no corridas**.

### Comparación vs LONG v3.5 (tip `bfc5713` / content `8e96fb8`)

| Exp | Antes (v3.5) | Ahora (v3.6) |
|---|---|---|
| E18 | NEGATIVE:2 NULL:3 PARTIAL:6 POSITIVE:4 STRONG:1 | NEGATIVE:2 NULL:3 PARTIAL:6 POSITIVE:4 STRONG:1 (**identical**) |
| E21 | NULL:1 PARTIAL:5 POSITIVE:1 STRONG:9 | NULL:1 PARTIAL:5 POSITIVE:1 **STRONG:9** (**identical**) |
| E22 | NEGATIVE:2 NULL:5 PARTIAL:3 POSITIVE:5 STRONG:1 | NEGATIVE:1 NULL:3 PARTIAL:3 **POSITIVE:8 STRONG:1** |
| E23 | PARTIAL:12 POSITIVE:2 STRONG:2 | PARTIAL:12 POSITIVE:2 STRONG:2 (**identical**; shared mix frozen) |
| E24 | NEGATIVE:4 NULL:2 PARTIAL:4 POSITIVE:6 | NEGATIVE:4 NULL:2 PARTIAL:4 POSITIVE:6 (**identical**) |
| E25 | NULL:3 PARTIAL:9 POSITIVE:4 | NULL:3 PARTIAL:9 POSITIVE:4 |
| E27 | NEGATIVE:2 NULL:1 PARTIAL:8 POSITIVE:5 | NEGATIVE:2 NULL:1 PARTIAL:8 POSITIVE:5 |
| E28 | PARTIAL:16 | PARTIAL:16 |

### Números reales (cuellos / ganancias)

- **E22**: e2e mean **0.770** (v3.5 0.674; **recovered past 0.72**); Relative-static mean ~0.305; **15/16** e2e>static; pt_err mean ~5.45; step1 ~0.95; oracle-mid ~0.78. Verdict: NEGATIVE 2→**1**, NULL 5→**3**, POSITIVE 5→**8**; PARTIAL+ = **12/16** (was 9/16). Mid-refine closed the decoder-OOD gap on A30A (NEG→PARTIAL) and lifted A301/A30B/A300/A303.
- **E23 horizons**: unchanged vs v3.5 mean cos h1..h64 = **[0.979, 0.950, 0.853, 0.684, 0.469, 0.325, 0.448]**; seeds ≥0.7: **[16,16,12,11,9,7,9]**. h8 preserved; h32/h64 still <0.7 mean (intentional: residual/local long TF diags lagged translation or hurt short).
- **E18/E21/E24/E25/E27**: POS/STRONG cluster **byte-identical** to v3.5 (shared `train_dynamics` frozen).
- **E28**: PARTIAL:16 mantenido.

### Gaps que quedan

1. E22 still NEGATIVE:1 (0xA30C hop-2/oracle-mid collapse) + NULL:3 (0xA302/A305/A309) — absolute e2e recovered but not every seed.
2. E23 h32/h64 still the open long-horizon gap; no safe lever found this cycle without short/E24 side effects.
3. Confirmation `0xB300–0xB30F` still deferred: E18 POS+STRONG only 5/16; E24 POS 6/16 — not stably majority POSITIVE/STRONG across E18/E21/E22/E24.

### Próximas palancas

1. E22: target 0xA30C hop-2 (oracle-mid 0.31) without raising Relative-static — selective T2 densification gated by mid-agree.
2. E23: horizon-conditioned residual that does **not** identity-lag translation (e.g. learned step-size), or accept residual gap.
3. Confirmation only once E18/E24 join E21/E22 in majority POSITIVE/STRONG.
