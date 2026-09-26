# Resultados — Etapa 2 Clean-Room v3 LONG

Protocolo: `docs/plan_autonomia_campo_v3.md` + `docs/etapa_2_autonomia_campo_experimentos.md` §29+.
Módulo: `src/field_autonomy_stage2_v2.rs`.
Rama: `exp/field-autonomy-next`.
Commit tip: `0b000b113107325e349611f3db99168fbb418a6b`.
Seeds: 16 (base DEV 0xA300..0xA307; extra=true)
Seeds confirmación `0xB300–0xB30F`: **corridas** — ver [`resultados_etapa_2_v3_confirm.md`](resultados_etapa_2_v3_confirm.md).
Periferia: numeric FIELD_ONLY; RQM/NN/table/attractor OFF en eval.

## Hardening en esta corrida (v3.7)

- `HyperparamLock::long()`: dyn_epochs=560 dyn_updates=7 (push absolute cos_dyn).
- **E18/E24 (v3.7)**: SoftScale train/dyn + Relative-only static probe (E21 dual-probe pattern).
- **E21 dual-probe retained**: SoftScale train/dyn + Relative-only static probe.
- **E22 (v3.7)**: mid-agree-gated closed-loop; latent-T2 hop gate else single-shot; disagree-lin∩mlp.
- **E23**: shared v3.5 mix retained (no h32/h64 lever this cycle).
- Dual FeatPath: SoftScale E18/E21/E22/E23/E24 dyn; Relative static probe E18/E21/E22/E24.
- Static baseline **sin** action + contraste geom-only; dyn extras + multi-familia + E21 dx denso.

Wall time: **1148.0s**. Filas: 240.

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
| `E18_reinforced_rule_learning` | NEGATIVE:1, NULL:1, POSITIVE:4, STRONG_POSITIVE:10 |
| `E19_provenance_audit` | NEGATIVE:1, POSITIVE:2, STRONG_POSITIVE:13 |
| `E20_rule_vs_trajectory_antimem` | PARTIAL:16 |
| `E21_interpolation_vs_extrapolation` | NULL:1, PARTIAL:5, POSITIVE:1, STRONG_POSITIVE:9 |
| `E22_composition_rqm_off` | NULL:2, PARTIAL:5, POSITIVE:8, STRONG_POSITIVE:1 |
| `E23_rollout_no_teacher_forcing` | PARTIAL:12, POSITIVE:2, STRONG_POSITIVE:2 |
| `E24_static_vs_dynamic_paired` | PARTIAL:2, POSITIVE:14 |
| `E25_central_experience_cdt_cycle` | NULL:3, PARTIAL:9, POSITIVE:4 |
| `E26_experience_ablation_A_G` | PARTIAL:16 |
| `E27_rule_transfer` | NEGATIVE:2, NULL:1, PARTIAL:8, POSITIVE:5 |
| `E28_continual_forgetting` | PARTIAL:16 |
| `E29_causal_intervention` | NULL:1, PARTIAL:15 |
| `E30_serialize_reload_persistence` | PARTIAL:16 |
| `FASE_A_cleanroom_field_only` | POSITIVE:16 |

CSV: [`resultados_etapa_2_v3_long.csv`](resultados_etapa_2_v3_long.csv)

## Lectura honesta (corrida LONG v3.7)

- Leakage FIELD_ONLY = **0**; contaminación DATASET_INVALID = **0**.
- `FASE_A` POSITIVE:16 (scaffold).
- Palancas v3.7: **E22 mid-agree-gated closed-loop + latent-T2 hop|shot fallback**; **E18/E24 SoftScale+Relative dual-probe** (E21 pattern); **E21/E23 unchanged**.
- Seeds confirmación `0xB300–0xB30F`: **justified and held** — see [`resultados_etapa_2_v3_confirm.md`](resultados_etapa_2_v3_confirm.md).

### Comparación vs LONG v3.6 (tip `ce282a2` / content `daa18dc`)

| Exp | Antes (v3.6) | Ahora (v3.7) |
|---|---|---|
| E18 | NEGATIVE:2 NULL:3 PARTIAL:6 POSITIVE:4 STRONG:1 (POS+STRONG 5/16) | NEGATIVE:1, NULL:1, POSITIVE:4, STRONG_POSITIVE:10 (**POS+STRONG 14/16**) |
| E21 | NULL:1 PARTIAL:5 POSITIVE:1 STRONG:9 | NULL:1, PARTIAL:5, POSITIVE:1, STRONG_POSITIVE:9 (**STRONG:9 protected**) |
| E22 | NEGATIVE:1 NULL:3 PARTIAL:3 POSITIVE:8 STRONG:1 (PARTIAL+ 12/16) | NULL:2, PARTIAL:5, POSITIVE:8, STRONG_POSITIVE:1 (**PARTIAL+ 14/16**; A30C NEG→PARTIAL) |
| E23 | PARTIAL:12 POSITIVE:2 STRONG:2 | PARTIAL:12, POSITIVE:2, STRONG_POSITIVE:2 (**identical**) |
| E24 | NEGATIVE:4 NULL:2 PARTIAL:4 POSITIVE:6 (POS 6/16) | PARTIAL:2, POSITIVE:14 (**POS+STRONG 14/16**) |
| E25 | NULL:3 PARTIAL:9 POSITIVE:4 | NULL:3, PARTIAL:9, POSITIVE:4 |
| E27 | NEGATIVE:2 NULL:1 PARTIAL:8 POSITIVE:5 | NEGATIVE:2, NULL:1, PARTIAL:8, POSITIVE:5 |
| E28 | PARTIAL:16 | PARTIAL:16 |

### Números reales (cuellos / ganancias)

- **E22**: e2e mean **0.822** (v3.6 0.770); Relative-static mean ~0.305; **16/16** e2e>static; oracle-mid mean ~0.748; step1 ~0.949. Verdict: NEGATIVE 1→**0**, NULL 3→**2**, PARTIAL+ 12→**14/16**. Seed 0xA30C hop oracle-mid still collapsed (~0.15) but mid-agree/latent-T2 gate routes to single-shot (PARTIAL). Residual NULL: 0xA305 (hop+shot both weak) + 0xA30B.
- **E18**: SoftScale dyn + Relative static → POS+STRONG **5→14/16**. Residual: 0xA304 NEGATIVE (dyn collapse) + 0xA309 NULL.
- **E24**: same dual-probe + full enc epochs → POS **6→14/16** (2 PARTIAL remain).
- **E21**: hist **byte-identical** to v3.6 (STRONG:9).
- **E23 horizons**: unchanged mean cos h1..h64 = **[0.979, 0.95, 0.853, 0.684, 0.469, 0.325, 0.448]**; seeds ≥0.7: **[16, 16, 12, 11, 9, 7, 9]**.

### Gaps que quedan

1. E22 NULL:2 (0xA305 absolute compose failure; 0xA30B hop-gated regression) — PARTIAL+ target met.
2. E18 still has 1 NEG + 1 NULL under SoftScale.
3. E23 h32/h64 still open (no safe lever; aborted).
4. Confirmation `0xB300–0xB30F`: **held** (E18 16/16, E21 12/16, E22 12/16, E24 16/16 POS+STRONG).

### Próximas palancas

1. E22: lift 0xA305 absolute compose (both paths weak) without raising Relative-static.
2. E18: diagnose SoftScale collapse on 0xA304.
3. E23: only if a lever does not hurt short horizons / POS cluster.
