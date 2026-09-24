# Resultados — Ciclo siguiente

**Rama:** `exp/mejoras-siguiente-ciclo`  
**Propuesta:** [`propuesta_mejoras_experimentos_ciclo_siguiente.md`](propuesta_mejoras_experimentos_ciclo_siguiente.md)  
**Base main:** `e05cef1`  
**Estado:** harness P0 **implementado**; **sin corridas numéricas PASS inventadas**.

---

## Implementado / pendiente de correr

| Ítem | Código | Corrida | Notas |
|---|---|---|---|
| P0.1 UI badges mode/seed | ✅ | n/a UI | Train + Tests badges |
| P0.2 `/api/tests/start` suite default=smoke | ✅ | n/a | `smoke` \| `experiments_smoke` \| `stage2_v2_dev` |
| P0.3 Telemetría leakage/field_only/rqm_eval | ✅ | n/a | status/events + panel |
| P0.4 Chat decoder-only | ✅ | test unitario | no escribe datasets |
| E21 harden (`0xA310–0xA317`) | ✅ harness | **pendiente** | `run_e21_harden` + example smoke |
| E22 harden (`0xA318–0xA31F`) | ✅ harness | **pendiente** | compose train-only auditado |
| E13 re-run 8 seeds `0xE1100–0xE1107` | protocol docs | **pendiente** | ver § E13 abajo |
| E15 suite 8 seeds | protocol docs | **pendiente** | ver § E15 abajo |
| E30-OS restart real | **diferido** | — | Sprint 3 |
| Confirmation `0xB300–0xB30F` | **prohibido** | **NO** | hasta lock |
| Gemma generative datasets | **diferido** | — | P2.2 docs only |
| Benches largos / DEV 8 full | **diferido** | — | CLI `--release` |

---

## Metadatos de corrida (rellenar tras suite real)

| Campo | Valor |
|---|---|
| Fecha (America/Mexico_City) | _TBD_ |
| Harness SHA | _TBD_ |
| Hyperparam lock id | sin retune; hereda Clean-Room DEV + harden edge curriculum |
| Seed family | `0xA310–0xA317` (E21) / `0xA318–0xA31F` (E22) |
| Confirmation tocada | **NO** |
| Wall time | _TBD_ |
| Filas CSV | _TBD_ |
| CSV | `docs/resultados_ciclo_siguiente.csv` _(crear al primera corrida)_ |

## Contaminación / leakage

| Check | Valor |
|---|---|
| Filas `DATASET_INVALID` | _TBD_ |
| Filas leakage>0 | _TBD_ (objetivo FIELD_ONLY = 0) |
| RQM/NN/table/attractor en brazo autonomía | **OFF** |

## Histogramas (rellenar tras suite — no inventar)

| Experimento | PASS | PARTIAL | NULL | NEGATIVE/FAIL | Notas |
|---|---:|---:|---:|---:|---|
| E13_holdout_8seed | | | | | pendiente |
| E15_compose_8seed | | | | | pendiente |
| E21_extrap_A31x | | | | | harness listo |
| E22_compose_rqm_off_A31x | | | | | harness listo |
| E30_os_restart | | | | | diferido |
| E31_sample_efficiency | | | | | diferido |
| E32_rollout_recovery | | | | | diferido |
| E33_ui_smoke | | | | | checklist producto |

## Producto (UI/API)

| Ítem P0/P1 | Landed | Evidencia |
|---|---|---|
| P0.1 badges smoke/dev | ✅ | `index.html` + `app.js` |
| P0.2 tests default smoke | ✅ | `parse_test_suite` + UI body `suite=smoke` |
| P0.3 telemetry leakage | ✅ | TestsJob / FieldEvalReport / panel |
| P0.4 Chat≠Train | ✅ | `decoder_only` + assert |
| P1.x … | diferido | |

## Protocolo corto — E13 re-run (`0xE1100–0xE1107`)

- **Harness:** `liquid_experiments_11_17::run_experiment_13` / `examples/smoke_e13_e15.rs`
- **Ablations:** −residual / −animal_ball / −tax_ancestors (documentar por seed)
- **Prohibido:** soft-target en cues holdout; insertar pares holdout
- **PASS:** unseen ≥0.50 y ≥2/3 holdout bits en ≥6/8; leakage=0
- **Estado:** pending run — no cifras aquí

## Protocolo corto — E15 suite (`0xE1100–0xE1107`)

- **Harness:** `run_experiment_15`
- **PASS:** structure 3/3 y rank≥0.70 en ≥6/8
- **Estado:** pending run — smoke 1-seed histórico no cuenta como suite

## CSV plantilla (crear al primera corrida)

Cabecera sugerida:

```text
experiment,seed,mode,field_only,rqm_eval,leakage,cos_dyn,cos_static,verdict,harness_sha,notes
```

## Honestidad

- [x] Ninguna cifra PASS inventada en este documento.
- [x] Confirmation no usada.
- [ ] CSV con provenance tras primera corrida real.
- [ ] Negativos reportados sin maquillaje (cuando existan).

## Decisión

| Pregunta | Respuesta |
|---|---|
| ¿Listo para confirmation `0xB300–0xB30F`? | **NO** |
| ¿Candidato a merge main? | **NO** (rama experimental) |
