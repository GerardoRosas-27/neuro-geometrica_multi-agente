# Resultados — Ciclo siguiente (plantilla)

**Rama:** `exp/mejoras-siguiente-ciclo`
**Propuesta:** [`propuesta_mejoras_experimentos_ciclo_siguiente.md`](propuesta_mejoras_experimentos_ciclo_siguiente.md)
**Base main:** `e05cef1`
**Estado:** **VACÍO — sin corridas todavía.** No inventar cifras.

---

## Metadatos de corrida (rellenar)

| Campo | Valor |
|---|---|
| Fecha (America/Mexico_City) | _TBD_ |
| Harness SHA | _TBD_ |
| Hyperparam lock id | _TBD_ (o “sin retune; hereda Clean-Room DEV”) |
| Seed family | _p.ej. 0xA310–0xA317_ |
| Confirmation tocada | **NO** / sí (`0xB300–0xB30F`) |
| Wall time | _TBD_ |
| Filas CSV | _TBD_ |
| CSV | `docs/resultados_ciclo_siguiente.csv` _(crear al primera corrida)_ |

## Contaminación / leakage

| Check | Valor |
|---|---|
| Filas `DATASET_INVALID` | _TBD_ |
| Filas leakage>0 | _TBD_ (objetivo FIELD_ONLY = 0) |
| RQM/NN/table/attractor en brazo autonomía | OFF / _desviar_ |

## Histogramas (rellenar tras suite)

| Experimento | PASS | PARTIAL | NULL | NEGATIVE/FAIL | Notas |
|---|---:|---:|---:|---:|---|
| E13_holdout_8seed | | | | | |
| E15_compose_8seed | | | | | |
| E21_extrap_A31x | | | | | |
| E22_compose_rqm_off_A31x | | | | | |
| E30_os_restart | | | | | |
| E31_sample_efficiency | | | | | |
| E32_rollout_recovery | | | | | |
| E33_ui_smoke | | | | | |

## Producto (UI/API)

| Ítem P0/P1 | Landed | Evidencia |
|---|---|---|
| P0.1 badges smoke/dev | | |
| P0.2 tests default smoke | | |
| P0.3 telemetry leakage | | |
| P0.4 Chat≠Train | | |
| P1.x … | | |

## Honestidad

- [ ] Ninguna cifra copiada de Clean-Room DEV sin re-corrida.
- [ ] Confirmation no usada para seleccionar arquitectura.
- [ ] CSV con provenance (seed, mode, field_only, harness SHA).
- [ ] Negativos reportados sin maquillaje.

## Decisión

| Pregunta | Respuesta |
|---|---|
| ¿Listo para confirmation `0xB300–0xB30F`? | _NO / sí + justificación_ |
| ¿Candidato a merge main? | _NO / sí + PR_ |
